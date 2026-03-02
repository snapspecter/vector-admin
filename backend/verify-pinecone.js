const { Pinecone } = require("./utils/vectordatabases/providers/pinecone/index.js");

async function checkPinecone() {
  const connector = {
    type: "pinecone",
    settings: {
      environment: "gcp-starter",
      index: "test-index",
      apiKey: "test-key"
    }
  };

  try {
    const pc = new Pinecone(connector);
    
    if (pc.name === "pinecone" && pc.config.settings.apiKey === "test-key") {
      console.log("SUCCESS: Pinecone provider instantiated successfully.");
      console.log("Config keys:", Object.keys(pc.config));
    } else {
      console.error("FAILED to instantiate Pinecone provider setup correctly.", pc);
    }
  } catch (e) {
    console.error("ERROR while orchestrating Pinecone:", e.message);
  }
}

checkPinecone();
