function draw_tagging(json_output, base) {
  const data = json_output["data"];
  console.log(data);
  base.selectAll("*").remove();
  const columns = base.selectAll("div").
  data(data).
  enter().
  append("div").
  classed("column", true);

  function updateColValues(key, spanned) {
    const task_class = key.split("_")[1];
    return columns.append("div").
    classed("col-value", true).
    attr("data-type", (d, i) => i == 0 ? key: null).
    classed(key, true).
    classed(task_class, d => {
      if (!spanned) return true;
      if (d[key] == "O") return false;
      return true;
    }).
    classed("spanned", d => {
      if (!spanned) return false;
      if (d[key] == "O") return false;
      return true;
    }).
    classed("span-start", d => {
      if (spanned && d[key][0] == "B") return true;
      return false;
    }).
    classed("span-inside", d => {
      if (spanned && d[key][0] == "I") return true;
      return false;
    }).
    text(d => {
      if (!spanned) return d[key];
      if (d[key] == "O" || d[key][0] == "I") return "\u00A0";
      return d[key].split("-").slice(1).join('-');
    });
  }

  const tokens = updateColValues("tokens", false);

  const ud_pos = updateColValues("ud_pos", false);
  const ark_pos = updateColValues("ark_pos", false);
  const ptb_pos = updateColValues("ptb_pos", false);

  const multimodal_ner = updateColValues("multimodal_ner", true);
  const broad_ner = updateColValues("broad_ner", true);
  const wnut17_ner = updateColValues("wnut17_ner", true);
  const ritter_ner = updateColValues("ritter_ner", true);
  const yodie_ner = updateColValues("yodie_ner", true);

  const ritter_chunk = updateColValues("ritter_chunk", true);
  const ritter_ccg = updateColValues("ritter_ccg", true);
}

function draw_classification(json_output, base) {
  const data = json_output;
  base.selectAll("*").remove();
  
  const data_entries = Object.entries(data)
    .filter(x => (x[0] != "text") && (x[0] != "tokens"))
    .map(x => { 
      return {
        obj_key: x[0], 
        parts: x[0].split("_"),
        values: Object.entries(x[1])
        .sort((a, b) => b[1]-a[1])
        .map((a, i) => [ ...a, i])
      }; 
    });
  console.log(data_entries);

  const tasks = d3.nest().key(d => d.parts[1]).entries(data_entries)
    .sort((a, b) => (a.key > b.key) ? 1 : -1);
  
  const task_divs = base.append("div")
    .classed("row", true)
    .selectAll("div.col-4")
    .data(tasks)
    .enter()
    .append("div")
    .classed("col-sm-4", true);
  
  const task_titles = task_divs
    .append("h2")
	  .text(d => d.key);
	
  const task_data_divs = task_divs
    .append("div")
    //.classed("card-columns", true)
    .selectAll("div.card")
    .data(d => d.values.sort((x, y) => x.parts[0].localeCompare(y.parts[0])))
    .enter()
    .append("div")
    .classed("card", true)
    .html(d => `<div class="card-header">${d.parts[0]}</div>`)
    .append("ul")
    .classed("list-group list-group-horizontal-sm", true)
  
	const label_prob_divs = task_data_divs.selectAll("li.list-group-item")
    .data(d => d.values.sort())
    .enter()
    .append("li")
    .classed("list-group-item", true)
    .html(d => {
      const badge = d[2] == 0 ? "danger" : "dark";
      return `${d[0].replace("_", " ")} <span class="badge badge-${badge}">${d[1].toFixed(3)}</span>`;
    });
  console.log(data);
}

function draw(json_output){
  console.log(json_output);
  draw_tagging(json_output["tagging"], base_tagging);
  draw_classification(json_output["classification"], base_classification);
  base_json.append("pre").text(JSON.stringify(json_output, undefined, 2));
}

const base = d3.select("#socialmediaie-output");
const base_classification = base.append("div").attr("id", "classification").classed("row", true).append("div").classed("col-lg-12", true);
const base_tagging = base.append("div").attr("id", "tagging").classed("row", true).append("div").classed("col-lg-12", true);
const base_json = base.append("div").attr("id", "json").classed("row", true).append("div").classed("col-lg-12", true);
const textarea = d3.select("#socialmediaie-input");
const predict_button = d3.select("form button");
const predict_button_spinner = predict_button.select("#buttonSpinner");
const predict_button_status = predict_button.select("#buttonStatus");
//const URL = "https://gist.githubusercontent.com/napsternxg/bd133f7ef4b31b684368dbfd832a9001/raw/0907aa70d1749823af3e3d871aa8450da916201f/SocialMediaIE_classifier_tagger_output.json";
const URL = "/predict_json"

async function drawFromInput(){
  const textInput = textarea.property('value');
  predict_button.attr("disabled", true);
  predict_button_spinner.style("visibility", "visible");
  predict_button_status.text("Loading ... ");
  const request_data = {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: "textInput=" + encodeURIComponent(textInput),
  };
  console.log(request_data);
  const json_output = await d3.json(URL, request_data);
  draw(json_output);
  predict_button.attr("disabled", null);
  predict_button_spinner.style("visibility", "hidden");
  predict_button_status.text("Predict");
}

predict_button.on("click", () => {
  d3.event.stopPropagation();
  d3.event.preventDefault();
  drawFromInput();
});

drawFromInput();