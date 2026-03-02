const { v4 } = require("uuid");
const { SystemSettings } = require("./systemSettings");

const Telemetry = {
  label: "telemetry_id",
  id: async function () {
    const result = await SystemSettings.get({ label: this.label });
    if (!!result?.value) return result.value;
    return result?.value;
  },
  client: function () {
    return null;
  },
  connect: async function () {
    const distinctId = await this.findOrCreateId();
    return { client: null, distinctId };
  },
  isDev: function () {
    return false;
  },
  sendTelemetry: async function (_event, _properties = {}, _force = false) {
    return;
  },
  flush: async function () {
    return;
  },
  setUid: async function () {
    const newId = v4();
    await SystemSettings.updateSettings({ [this.label]: newId });
    return newId;
  },
  findOrCreateId: async function () {
    const currentId = await this.id();
    if (!!currentId) return currentId;
    const newId = await this.setUid();
    return newId;
  },
};

module.exports = { Telemetry };
