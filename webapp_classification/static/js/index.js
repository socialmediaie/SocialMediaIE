function draw(json_output) {
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
    
  base.append("pre").text(JSON.stringify(data, undefined, 2));
  console.log(data);
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