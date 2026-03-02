const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { OpenAi } = require("../../../openAi");
const { v4 } = require("uuid");
const { DocumentVectors } = require("../../../../models/documentVectors");
const { toChunks } = require("../../utils");
const { storeVectorResult } = require("../../../storage");
const { WorkspaceDocument } = require("../../../../models/workspaceDocument");

class Pinecone {
  constructor(connector) {
    this.name = "pinecone";
    this.config = this.setConfig(connector);
    this.STARTER_TIER_UPSERT_DELAY = 15_000;
  }

  setConfig(config) {
    var { type, settings } = config;
    if (typeof settings === "string") settings = JSON.parse(settings);
    return { type, settings };
  }

  // Docs: https://docs.pinecone.io/docs/projects#project-environment
  // This tier does not allow namespace creation.
  isStarterTier() {
    const { settings } = this.config;
    return settings.environment === "gcp-starter";
  }

  async connect() {
    const { Pinecone } = require("@pinecone-database/pinecone");
    const { type, settings } = this.config;

    if (type !== "pinecone")
      throw new Error("Pinecone::Invalid Not a Pinecone connector instance.");

    const client = new Pinecone({
      apiKey: settings.apiKey,
    });

    const pineconeIndex = client.index(settings.index);
    
    let ready = false;
    let host = null;
    try {
      const model = await client.describeIndex(settings.index);
      ready = model.status.state === "Ready";
      host = model.host;
    } catch (e) {
      console.error("Pinecone.describeIndex", e);
    }

    if (!ready) throw new Error("Pinecone::Index not ready.");

    return { client, host, pineconeIndex };
  }

  async indexDimensions() {
    const { pineconeIndex } = await this.connect();
    const description = await pineconeIndex.describeIndexStats();
    return Number(description?.dimension || 0);
  }


  async totalIndicies() {
    const { pineconeIndex } = await this.connect();
    const { namespaces } = await pineconeIndex.describeIndexStats();
    const totalVectors = Object.values(namespaces || {}).reduce(
      (a, b) => a + (b?.recordCount || 0),
      0,
    );
    return { result: totalVectors, error: null };
  }

  // Collections === namespaces for Pinecone to normalize interfaces
  async collections() {
    return await this.namespaces();
  }

  async namespaces() {
    const { pineconeIndex } = await this.connect();
    const { namespaces } = await pineconeIndex.describeIndexStats();
    const collections = Object.entries(namespaces || {}).map(
      ([name, values]) => {
        return {
          name,
          count: values?.recordCount || 0,
        };
      },
    );

    return collections;
  }

  async namespaceExists(index, namespace = null) {
    if (namespace === null) throw new Error("No namespace value provided.");
    const { namespaces } = await index.describeIndexStats();
    return (namespaces || {}).hasOwnProperty(namespace);
  }

  async namespace(name = null) {
    if (name === null) throw new Error("No namespace value provided.");
    const { pineconeIndex } = await this.connect();
    const { namespaces } = await pineconeIndex.describeIndexStats();
    const collection = (namespaces || {}).hasOwnProperty(name)
      ? namespaces[name]
      : null;
    if (!collection) return null;

    return {
      name,
      ...collection,
      count: collection?.recordCount || 0,
    };
  }

  // 3-try topK progressive backoff when error occurs.
  // This helps when the associated text/metadata exceeds the max POST response size Pinecone is willing to send.
  async rawQuery(pineconeIndex, namespace, queryParams = {}) {
    const initialPageSize = queryParams?.topK || 1_000;
    try {
      return await pineconeIndex.namespace(namespace).query(queryParams);
    } catch (e) {
      try {
        return await pineconeIndex.namespace(namespace).query({
          ...queryParams,
          topK: Math.floor(initialPageSize / 2),
        });
      } catch (e2) {
        try {
          return await pineconeIndex.namespace(namespace).query({
            ...queryParams,
            topK: Math.floor(initialPageSize / 4),
          });
        } catch (e3) {
          console.error("Pinecone.rawQuery", e3.message);
          return { matches: [], error: e3.message };
        }
      }
    }
  }

  async rawGet(host, namespace, offset = 10, filterRunId = "") {
    try {
      const data = {
        ids: [],
        embeddings: [],
        metadatas: [],
        documents: [],
        error: null,
      };

      const { pineconeIndex } = await this.connect();
      const dimension = await this.indexDimensions();
      const dummyVector = Array.from({ length: dimension }, () => 0.0001);

      const queryRequest = {
        topK: offset,
        includeValues: true,
        includeMetadata: true,
        vector: dummyVector,
      };

      if (filterRunId) {
        queryRequest.filter = { runId: { $ne: filterRunId } };
      }

      const queryResult = await this.rawQuery(pineconeIndex, namespace, queryRequest);
      if (!queryResult?.matches || queryResult.matches.length === 0) {
        return { ...data, error: queryResult?.error || null };
      }

      queryResult.matches.forEach((match) => {
        const { id, values = [], metadata = {} } = match;
        data.ids.push(id);
        data.embeddings.push(values);
        data.metadatas.push(metadata);
        data.documents.push(metadata?.text ?? "");
      });
      return data;
    } catch (error) {
      console.error("Pinecone::RawGet", error.message);
      return {
        ids: [],
        embeddings: [],
        metadatas: [],
        documents: [],
        error,
      };
    }
  }

  // Split, embed, and save a given document data that we get from the document processor
  // API.
  async processDocument(
    namespace,
    documentData,
    embedderApiKey,
    dbDocument,
    pineconeIndex,
  ) {
    try {
      const openai = new OpenAi(embedderApiKey);
      const { pageContent, id, ...metadata } = documentData;
      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 20,
      });
      const textChunks = await textSplitter.splitText(pageContent);

      console.log("Chunks created from document:", textChunks.length);
      const documentVectors = [];
      const cacheInfo = [];
      const vectors = [];
      const vectorValues = await openai.embedTextChunks(textChunks);
      const submission = {
        ids: [],
        embeddings: [],
        metadatas: [],
        documents: [],
      };

      if (!!vectorValues && vectorValues.length > 0) {
        for (const [i, vector] of vectorValues.entries()) {
          const vectorRecord = {
            id: v4(),
            values: vector,
            // [DO NOT REMOVE]
            // LangChain will be unable to find your text if you embed manually and dont include the `text` key.
            // https://github.com/hwchase17/langchainjs/blob/2def486af734c0ca87285a48f1a04c057ab74bdf/langchain/src/vectorstores/pinecone.ts#L64
            metadata: { ...metadata, text: textChunks[i] },
          };

          submission.ids.push(vectorRecord.id);
          submission.embeddings.push(vectorRecord.values);
          submission.metadatas.push(metadata);
          submission.documents.push(textChunks[i]);

          vectors.push(vectorRecord);
          documentVectors.push({
            docId: id,
            vectorId: vectorRecord.id,
            documentId: dbDocument.id,
            workspaceId: dbDocument.workspace_id,
            organizationId: dbDocument.organization_id,
          });
          cacheInfo.push({
            vectorDbId: vectorRecord.id,
            values: vector,
            metadata: vectorRecord.metadata,
          });
        }
      } else {
        console.error(
          "Could not use OpenAI to embed document chunk! This document will not be recorded.",
        );
      }

      if (vectors.length > 0) {
        const chunks = [];
        for (const chunk of toChunks(vectors, 500)) {
          chunks.push(chunk);
          await pineconeIndex.namespace(namespace).upsert(chunk);
        }
      }

      await DocumentVectors.createMany(documentVectors);
      await storeVectorResult(
        cacheInfo,
        WorkspaceDocument.vectorFilename(dbDocument),
      );
      return { success: true, message: null };
    } catch (e) {
      console.error("addDocumentToNamespace", e.message);
      return { success: false, message: e.message };
    }
  }

  async similarityResponse(namespace, queryVector, topK = 4) {
    const { pineconeIndex } = await this.connect();
    const result = {
      vectorIds: [],
      contextTexts: [],
      sourceDocuments: [],
      scores: [],
    };
    const response = await pineconeIndex.namespace(namespace).query({
      vector: queryVector,
      topK,
      includeMetadata: true,
    });

    response.matches.forEach((match) => {
      result.vectorIds.push(match.id);
      result.contextTexts.push(match.metadata.text);
      result.sourceDocuments.push(match);
      result.scores.push(match.score);
    });

    return result;
  }

  async getMetadata(namespace = "", vectorIds = []) {
    const { pineconeIndex } = await this.connect();
    const { records } = await pineconeIndex
      .namespace(namespace)
      .fetch(vectorIds);
    const metadatas = [];

    Object.values(records || {})?.forEach((vector, i) => {
      metadatas.push({
        vectorId: vector.id,
        ...(vector?.metadata || {}),
      });
    });

    return metadatas;
  }
}

module.exports.Pinecone = Pinecone;
