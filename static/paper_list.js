'use strict';

const Paper = props => {
  const p = props.paper
  return (
    <div class={'rel_paper ' + p.Number}>
      <div class='dllinks'>
        <div class='metadata rel_date'>{p.Number}</div>
      </div>
      <div class='rel_title'>{p.adikaram_name}</div>
      <div class='rel_authors'>{p.kural}</div>
      <div class='rel_abs'>{p.mk}</div>
      <div class='rel_abs'>{p.mv}</div>
      <div class='rel_abs'>{p.sp}</div>
    </div>
  )
}

const PaperList = props => {
  const lst = props.papers;
  const plst = lst.map((jpaper, ix) => <Paper key={ix} paper={jpaper} />);
  const msg = {
    "latest": "Showing few kurals for the day:",
    "sim": 'Showing papers most similar to the first one:',
    "search": 'Search results for "' + gvars.search_query + '":'
  }
  return (
    <div>
      <div id="info">{msg[gvars.sort_order]}</div>
      <div id="paperList" class="rel_papers">
        {plst}
      </div>
    </div>
  )
}

ReactDOM.render(<PaperList papers={papers} />, document.getElementById('wrap'));
