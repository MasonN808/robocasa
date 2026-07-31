#!/usr/bin/env python
"""Regenerates the v3-matrix results artifact from whatever metrics exist.

Safe to re-run at any time: cells with no results yet render as "pending"
rather than being omitted, so the page can be published early and refreshed
as jobs land.

  python build_v3_artifact.py --out /path/to/page.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path("training/bc_task_vlm/eval_runs")
SOURCES = [("vs", "verbalized_sampling"), ("sr", "structured_random")]
OBS = [("cent", "centralized"), ("partial", "partial-obs")]
REASON = [("noreason", "no reasoning"), ("reason", "reasoning")]
SPLITS = [("heldout_trajectories", "seen tasks"), ("heldout_tasks", "unseen tasks")]


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def collect() -> list[dict]:
    rows = []
    for s_tag, s_name in SOURCES:
        for o_tag, o_name in OBS:
            for r_tag, r_name in REASON:
                # Centralized cells are sourced only from the "-notc" runs. The
                # earlier centralized models were trained with a task_complete
                # tool that partial never had, so their terminations and their
                # sample counts were not comparable to anything else here.
                # Those runs still exist on disk but are deliberately not read;
                # a centralized cell renders "pending" until its
                # no-task_complete replacement lands.
                suffix = "-notc" if o_tag == "cent" else ""
                cell = f"{s_tag}-{o_tag}-{r_tag}{suffix}"
                for sp_tag, sp_name in SPLITS:
                    off = _load(
                        RUNS / f"offsim_qwen3vl-8b-v3-{cell}__{sp_tag}"
                        / "structured_eval_metrics.json"
                    )
                    jud = _load(
                        RUNS / f"offsim_qwen3vl-8b-v3-{cell}__{sp_tag}"
                        / "comm_judge_metrics.json"
                    )
                    live = _load(
                        RUNS / f"livesim_qwen3vl-8b-v3-{cell}__{sp_tag}"
                        / "live_sim_metrics.json"
                    )
                    rows.append(
                        {
                            "source": s_name,
                            "obs": o_name,
                            "reasoning": r_name,
                            "split": sp_name,
                            "cell": cell,
                            # off-sim
                            "act_exact": off and off.get(
                                "structured_eval_action_exact_call_accuracy"),
                            "act_tool": off and off.get(
                                "structured_eval_action_tool_name_accuracy"),
                            "valid": off and off.get(
                                "structured_eval_tool_call_valid_rate"),
                            "n_samples": off and off.get("structured_eval_num_samples"),
                            "comm_raw": off and off.get(
                                "structured_eval_comm_exact_call_accuracy"),
                            # judged
                            "comm_judged": jud and jud.get("comm_judged_match_rate"),
                            "judged_overall": jud and jud.get(
                                "judged_exact_call_accuracy"),
                            # live-sim
                            "fsm": live and live.get("fsm_goal_rate"),
                            "native": live and live.get("native_success_rate"),
                            "harness_err": live and live.get("harness_error_rate"),
                            "n_traj": live and live.get("num_trajectories"),
                            "terminations": live and live.get("terminations"),
                        }
                    )
    return rows


BASELINES = [("gemini-3-flash-preview", "Gemini 3 Flash")]


def collect_baselines() -> list[dict]:
    """Off-the-shelf models run through the same manifests, splits and tools.

    These are not cells of the matrix -- nothing was trained -- so they are
    carried separately and drawn as a reference line rather than compared
    head-to-head. One caveat travels with every number here: an untrained
    model is met in its own dialect (native function calls) while the tuned
    cells are met in theirs (JSON in text), because forcing an untrained
    model into the trained dialect scores 0 for format reasons alone.
    """

    out = []
    for model_id, model_label in BASELINES:
        tag = model_id.replace(".", "-").lower()
        for s_tag, s_name in SOURCES:
            for o_tag, o_name in OBS:
                for sp_tag, sp_name in SPLITS:
                    stem = f"ots-{tag}-{o_tag}-{s_tag}__{sp_tag}"
                    off = _load(
                        RUNS / f"offsim_{stem}" / "structured_eval_metrics.json"
                    )
                    live = _load(RUNS / f"livesim_{stem}" / "live_sim_metrics.json")
                    out.append(
                        {
                            "model": model_label,
                            "source": s_name,
                            "obs": o_name,
                            "split": sp_name,
                            "act_exact": off and off.get(
                                "structured_eval_action_exact_call_accuracy"),
                            "act_tool": off and off.get(
                                "structured_eval_action_tool_name_accuracy"),
                            "valid": off and off.get(
                                "structured_eval_tool_call_valid_rate"),
                            "comm_raw": off and off.get(
                                "structured_eval_comm_exact_call_accuracy"),
                            "n_samples": off and off.get(
                                "structured_eval_num_samples"),
                            "fsm": live and live.get("fsm_goal_rate"),
                            "n_traj": live and live.get("num_trajectories"),
                            "judged_overall": None,
                            "comm_judged": None,
                            "reasoning": "untrained",
                        }
                    )
    return out


def render(rows: list[dict], base: list[dict]) -> str:
    done_off = sum(1 for r in rows if r["act_exact"] is not None)
    done_live = sum(1 for r in rows if r["fsm"] is not None)
    done_jud = sum(1 for r in rows if r["judged_overall"] is not None)
    data = json.dumps(rows)
    return (
        HTML_HEAD
        + f"<script>const ROWS={data};const BASE={json.dumps(base)};"
        + f"const PROG={{off:{done_off},live:{done_live},jud:{done_jud},total:{len(rows)}}};</script>"
        + HTML_BODY
    )


HTML_HEAD = """<title>v3 matrix — off-sim &amp; live-sim</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root{color-scheme:light;
    --bg:#f4f6f7; --card:#ffffff; --soft:#eaeef1; --line:#dce2e6; --line2:#c2ccd2;
    --ink:#11161a; --ink2:#4f5a62; --ink3:#87939a;
    --a:#2a78d6; --a-soft:#cde2fb; --b:#eb6834; --b-soft:#fbdccb;
    --good:#0ca30c; --good-soft:#e0f3e0; --bad:#d03b3b; --bad-soft:#fbe4e4;
    --mono:ui-monospace,"SF Mono","Cascadia Code",Consolas,monospace;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
  @media(prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;
    --bg:#0e1012;--card:#15181b;--soft:#1e2226;--line:#2a2f34;--line2:#3a4046;
    --ink:#f1f4f6;--ink2:#b4bbc1;--ink3:#7b848b;
    --a:#3987e5;--a-soft:#1b3a5b;--b:#d95926;--b-soft:#482b19;
    --good:#2fc22f;--good-soft:#153a1b;--bad:#e66767;--bad-soft:#3f1f1f;}}
  :root[data-theme="dark"]{color-scheme:dark;
    --bg:#0e1012;--card:#15181b;--soft:#1e2226;--line:#2a2f34;--line2:#3a4046;
    --ink:#f1f4f6;--ink2:#b4bbc1;--ink3:#7b848b;
    --a:#3987e5;--a-soft:#1b3a5b;--b:#d95926;--b-soft:#482b19;
    --good:#2fc22f;--good-soft:#153a1b;--bad:#e66767;--bad-soft:#3f1f1f;}
  :root[data-theme="light"]{color-scheme:light;
    --bg:#f4f6f7;--card:#ffffff;--soft:#eaeef1;--line:#dce2e6;--line2:#c2ccd2;
    --ink:#11161a;--ink2:#4f5a62;--ink3:#87939a;
    --a:#2a78d6;--a-soft:#cde2fb;--b:#eb6834;--b-soft:#fbdccb;
    --good:#0ca30c;--good-soft:#e0f3e0;--bad:#d03b3b;--bad-soft:#fbe4e4;}
  *{box-sizing:border-box}html,body{margin:0;padding:0}
  body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased}
  .wrap{max-width:760px;margin:0 auto;padding:20px 16px 64px;display:flex;flex-direction:column;gap:20px}
  .eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3)}
  h1{font-family:var(--mono);font-size:21px;font-weight:700;margin:0;letter-spacing:-.01em;text-wrap:balance}
  .dek{font-size:13.5px;color:var(--ink2)}
  code{font-family:var(--mono);font-size:.92em;background:var(--soft);padding:1px 5px;border-radius:3px}
  header{display:flex;flex-direction:column;gap:8px;padding-bottom:16px;border-bottom:1px solid var(--line)}
  .prog{display:flex;flex-wrap:wrap;gap:8px}
  .pchip{font-family:var(--mono);font-size:11.5px;font-weight:700;padding:4px 9px;border-radius:6px;background:var(--soft);color:var(--ink2)}
  .pchip b{color:var(--ink)}
  .controls{display:flex;flex-direction:column;gap:10px}
  .clabel{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink3)}
  .seg{display:flex;background:var(--soft);border-radius:9px;padding:3px;gap:3px}
  .seg button{flex:1;font-family:var(--mono);font-size:12.5px;font-weight:600;border:0;background:transparent;color:var(--ink2);padding:9px 8px;border-radius:6px;cursor:pointer}
  .seg button.on{background:var(--card);color:var(--ink);box-shadow:0 1px 2px rgba(0,0,0,.06)}
  .seg button:focus-visible{outline:2px solid var(--a);outline-offset:1px}
  section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;display:flex;flex-direction:column;gap:4px}
  h2{font-family:var(--mono);font-size:14px;font-weight:700;margin:0}
  .note{font-size:12px;color:var(--ink3);margin-top:2px}
  .grp{padding:12px 0;border-top:1px solid var(--line)}
  .grp:first-of-type{border-top:0;padding-top:4px}
  .gtitle{font-family:var(--mono);font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--ink3);margin-bottom:8px}
  .row{display:flex;align-items:center;gap:9px;margin:6px 0}
  .rlab{font-size:12.5px;width:104px;flex:none;color:var(--ink2)}
  .track{flex:1;height:13px;background:var(--soft);border-radius:4px;overflow:hidden}
  .fill{height:100%;border-radius:4px 0 0 4px}
  .fill.a{background:var(--a)}.fill.b{background:var(--b)}
  /* untrained reference: a floor to clear, not another cell */
  .fill.base{background:repeating-linear-gradient(45deg,var(--line2),var(--line2) 3px,transparent 3px,transparent 6px);border:1px solid var(--line2)}
  .rlab.base,.val.base{color:var(--ink3);font-style:italic}
  .val{font-family:var(--mono);font-size:12.5px;font-variant-numeric:tabular-nums;width:56px;text-align:right;flex:none}
  .val.pend{color:var(--ink3)}
  .delta{display:flex;justify-content:flex-end;margin-top:2px}
  .dchip{font-family:var(--mono);font-size:11px;font-weight:700;padding:2px 7px;border-radius:5px}
  .dchip.up{background:var(--good-soft);color:var(--good)}
  .dchip.dn{background:var(--bad-soft);color:var(--bad)}
  .dchip.na{background:var(--soft);color:var(--ink3)}
  .callout{background:var(--soft);border-radius:9px;padding:12px 14px;font-size:12.5px;color:var(--ink2);line-height:1.6}
  .callout b{color:var(--ink)}
  details{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
  summary{cursor:pointer;font-family:var(--mono);font-size:13px;font-weight:700;list-style:none;display:flex;gap:8px;align-items:center}
  summary::-webkit-details-marker{display:none}
  summary::before{content:"▸";color:var(--ink3)}
  details[open] summary::before{content:"▾"}
  .scroll{overflow-x:auto;margin-top:10px;border:1px solid var(--line);border-radius:8px;-webkit-overflow-scrolling:touch}
  table{border-collapse:collapse;width:100%;min-width:720px;font-size:12px}
  th,td{padding:6px 9px;text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
  th:nth-child(-n+4),td:nth-child(-n+4){text-align:left}
  thead th{background:var(--soft);font-size:10px;text-transform:uppercase;color:var(--ink2);position:sticky;top:0}
  tbody tr:nth-child(even){background:var(--bg)}
  footer{font-family:var(--mono);font-size:11px;color:var(--ink3);border-top:1px solid var(--line);padding-top:14px;display:flex;flex-wrap:wrap;gap:4px 16px}
</style>
"""

HTML_BODY = """
<div class="wrap">
  <header>
    <div class="eyebrow">RoboCasa · bc_task_vlm · Qwen3-VL-8B LoRA</div>
    <h1>v3 matrix: data source × observability × reasoning</h1>
    <p class="dek">8 trained cells, each evaluated two ways — <b>off-sim</b> (teacher-forced next-step
    agreement) and <b>live-sim</b> (closed-loop rollouts in MuJoCo, the metric that reflects actually
    finishing the task). Communication is scored by LLM judge, since raw exact-match on free text
    measures phrasing conventions rather than quality.</p>
    <div class="prog" id="prog"></div>
  </header>

  <div class="controls">
    <div><div class="clabel">Metric</div><div class="seg" id="mseg"></div></div>
    <div><div class="clabel">Split</div><div class="seg" id="sseg"></div></div>
  </div>

  <section>
    <h2 id="mtitle"></h2>
    <p class="note" id="mnote"></p>
    <div id="chart"></div>
  </section>

  <div class="callout" id="caveat"></div>

  <details>
    <summary>All cells, every metric</summary>
    <div class="scroll"><table id="tbl"></table></div>
  </details>

  <footer id="foot"></footer>
</div>
<script>
const METRICS=[
 {k:"fsm",label:"Live-sim task success (FSM)",note:"Closed-loop: fraction of 75 rollouts where the task spec's symbolic goal was satisfied. The bar that matters most.",pct:1},
 {k:"judged_overall",label:"Judged overall (off-sim)",note:"Action exact-match + LLM-judged communication, over all steps.",pct:1},
 {k:"comm_judged",label:"Judged communication (off-sim)",note:"Paraphrase-tolerant judge score on communicate steps only.",pct:1},
 {k:"act_exact",label:"Action exact-match (off-sim)",note:"Physical steps only, exact call match.",pct:1},
 {k:"act_tool",label:"Correct tool selected (off-sim)",note:"Right tool chosen, regardless of arguments.",pct:1},
 {k:"valid",label:"Valid tool-call rate (off-sim)",note:"Predictions that parsed into a schema-valid call.",pct:1},
];
let M=METRICS[0].k, S="unseen tasks";
const $=(id)=>document.getElementById(id);
const fmt=(v)=>v===null||v===undefined?"—":(v*100).toFixed(1)+"%";
const fmtD=(v)=>{const p=v*100;return (p>0?"+":p<0?"−":"±")+Math.abs(p).toFixed(1)+"pp";};

function prog(){
  $("prog").innerHTML=
   `<span class="pchip">off-sim <b>${PROG.off}</b>/${PROG.total}</span>`+
   `<span class="pchip">judged <b>${PROG.jud}</b>/${PROG.total}</span>`+
   `<span class="pchip">live-sim <b>${PROG.live}</b>/${PROG.total}</span>`;
}
function segs(){
  $("mseg").innerHTML="";
  METRICS.forEach(m=>{const b=document.createElement("button");b.textContent=m.label.replace(/ \\(off-sim\\)| \\(FSM\\)/,"");
    b.className=m.k===M?"on":"";b.onclick=()=>{M=m.k;draw();};$("mseg").appendChild(b);});
  $("mseg").style.flexWrap="wrap";
  $("sseg").innerHTML="";
  ["seen tasks","unseen tasks"].forEach(s=>{const b=document.createElement("button");b.textContent=s;
    b.className=s===S?"on":"";b.onclick=()=>{S=s;draw();};$("sseg").appendChild(b);});
}
function draw(){
  segs();
  const m=METRICS.find(x=>x.k===M);
  $("mtitle").textContent=m.label; $("mnote").textContent=m.note;
  const host=$("chart"); host.innerHTML="";
  let max=0; ROWS.forEach(r=>{if(r.split===S&&r[M]!=null&&r[M]>max)max=r[M];});
  max=Math.max(max*1.1,0.05);
  ["verbalized_sampling","structured_random"].forEach(src=>{
    const g=document.createElement("div"); g.className="grp";
    const t=document.createElement("div"); t.className="gtitle"; t.textContent="trained on "+src; g.appendChild(t);
    ["centralized","partial-obs"].forEach(obs=>{
      const pair=["no reasoning","reasoning"].map(rz=>ROWS.find(r=>r.source===src&&r.obs===obs&&r.reasoning===rz&&r.split===S));
      pair.forEach((r,i)=>{
        const row=document.createElement("div"); row.className="row";
        const lab=document.createElement("div"); lab.className="rlab";
        lab.textContent=obs+(i?" +reason":"");
        const tr=document.createElement("div"); tr.className="track";
        const fl=document.createElement("div"); fl.className="fill "+(i?"b":"a");
        const v=r?r[M]:null;
        fl.style.width=(v==null?0:Math.max(2,v/max*100))+"%"; tr.appendChild(fl);
        const val=document.createElement("div"); val.className="val"+(v==null?" pend":""); val.textContent=v==null?"pending":fmt(v);
        row.appendChild(lab);row.appendChild(tr);row.appendChild(val); g.appendChild(row);
      });
      // Untrained reference for this same source/observability/split. Drawn
      // muted and outlined so it reads as a floor, not a fifth cell.
      const bs=BASE.find(b=>b.source===src&&b.obs===obs&&b.split===S);
      if(bs&&bs[M]!=null){
        const row=document.createElement("div"); row.className="row";
        const lab=document.createElement("div"); lab.className="rlab base";
        lab.textContent=bs.model+" (untrained)";
        const tr=document.createElement("div"); tr.className="track";
        const fl=document.createElement("div"); fl.className="fill base";
        fl.style.width=Math.max(2,bs[M]/max*100)+"%"; tr.appendChild(fl);
        const val=document.createElement("div"); val.className="val base";
        val.textContent=fmt(bs[M]);
        row.appendChild(lab);row.appendChild(tr);row.appendChild(val); g.appendChild(row);
      }
      if(pair[0]&&pair[1]&&pair[0][M]!=null&&pair[1][M]!=null){
        const d=pair[1][M]-pair[0][M];
        const w=document.createElement("div"); w.className="delta";
        const c=document.createElement("span");
        c.className="dchip "+(d>0.0005?"up":d<-0.0005?"dn":"na");
        c.textContent="reasoning "+fmtD(d); w.appendChild(c); g.appendChild(w);
      }
    });
    host.appendChild(g);
  });
  caveat();
  table();
}
function caveat(){
  const c=$("caveat");
  if(M==="act_exact"||M==="act_tool"||M==="valid"){
    c.innerHTML="<b>Centralized and partial evaluate different step populations.</b> Under "+
      "<code>--predict-acting-agent</code> the centralized cells score every <code>get_image</code> call as a "+
      "prediction target, so they are graded over substantially more samples than partial — and <code>get_image</code> "+
      "is an easier prediction than a physical action. Treat centralized-vs-partial gaps on these off-sim metrics as "+
      "inflated until recomputed over the shared step set. (task_complete is no longer trained or scored anywhere.)"+
      "<br><br><b>The untrained reference is met in its own dialect.</b> Gemini answers with native function calls; "+
      "the tuned cells answer with JSON in text, as trained. Forcing an untrained model into the trained dialect "+
      "scores 0 for format reasons alone, so this is the fair reading — but it is not a dialect-controlled comparison.";
  } else if(M==="fsm"){
    c.innerHTML="<b>This is the metric to trust for capability.</b> It measures whether the task was actually "+
      "completed in simulation, not whether the next token matched a demonstration. Cells can score ~0.98 on off-sim "+
      "action exact-match and still finish only a fraction of episodes.";
  } else {
    c.innerHTML="<b>Judged, not exact-match.</b> Raw exact-match on communication measures how closely a model "+
      "reproduces its training source's phrasing — a cross-source check measured up to a 17× own-vs-other gap on "+
      "that metric alone. These numbers come from an LLM judge that accepts paraphrase.";
  }
}
const COLS=[["source","source"],["obs","observability"],["reasoning","reasoning"],["split","split"],
 ["fsm","live fsm"],["judged_overall","judged all"],["comm_judged","judged comm"],
 ["act_exact","act exact"],["act_tool","act tool"],["valid","valid"],["comm_raw","raw comm"],
 ["n_samples","n samp"],["n_traj","n traj"]];
function table(){
  const t=$("tbl"); t.innerHTML="";
  const th=document.createElement("thead"); const tr=document.createElement("tr");
  COLS.forEach(([,l])=>{const e=document.createElement("th");e.textContent=l;tr.appendChild(e);});
  th.appendChild(tr); t.appendChild(th);
  const tb=document.createElement("tbody");
  ROWS.forEach(r=>{const row=document.createElement("tr");
    COLS.forEach(([k])=>{const td=document.createElement("td");
      const v=r[k];
      td.textContent=(v==null)?"—":(typeof v==="number"&&v<=1&&["fsm","judged_overall","comm_judged","act_exact","act_tool","valid","comm_raw"].includes(k))?(v*100).toFixed(1):v;
      row.appendChild(td);});
    tb.appendChild(row);});
  t.appendChild(tb);
}
$("foot").innerHTML="<span>judge: gemini-3-flash-preview</span><span>75 trajectories per live-sim split</span>"+
  "<span>effective batch 64, 3 epochs, LoRA r=16</span><span>pending cells refresh on republish</span>";
prog(); draw();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    rows = collect()
    base = collect_baselines()
    args.out.write_text(render(rows, base), encoding="utf-8")
    off = sum(1 for r in rows if r["act_exact"] is not None)
    live = sum(1 for r in rows if r["fsm"] is not None)
    jud = sum(1 for r in rows if r["judged_overall"] is not None)
    print(f"wrote {args.out}  (off-sim {off}/{len(rows)}, judged {jud}/{len(rows)}, live-sim {live}/{len(rows)})")


if __name__ == "__main__":
    main()
