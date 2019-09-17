function draw(json_output) {
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

const base = d3.select("#socialmediaie-output");
const textarea = d3.select("#textInput");
const predict_button = d3.select("form button");
const predict_button_spinner = predict_button.select("#buttonSpinner");
const predict_button_status = predict_button.select("#buttonStatus");
//const URL = "https://gist.githubusercontent.com/napsternxg/a9946616013e61d51d207b030a7fd1b3/raw/04ab99138f5750894357fcbb271397ca20a6f710/SocialMediaIE_tagger_output.json";
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