// Runs the browser inference logic (copied from docs/index.html) in Node and prints probabilities as JSON.
// usage: node tests/parity.js '["review 1","review 2"]'
const fs = require("fs"), path = require("path");
const html = fs.readFileSync(path.join(__dirname, "..", "docs", "index.html"), "utf8");
const src = html.match(/function localPredict[\s\S]*?\n  }\n/)[0];
const m = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "docs", "model.json"), "utf8"));
m.stop = new Set(m.stop_words);
const localPredict = eval("(" + src.replace("function localPredict", "function") + ")");
console.log(JSON.stringify(JSON.parse(process.argv[2]).map(t => localPredict(m, t).prob_positive)));
