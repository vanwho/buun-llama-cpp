#!/usr/bin/env python3
"""Rebuild the 17 article/gallery charts from the bundled curated chart-data.csv.

Requires Python 3.10+ and matplotlib. Reads only ./data/chart-data.csv;
writes PNG/SVG into ./figures and captions to ./data/chart_captions.json.
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "figures"
RESEARCH = ROOT / "data"
PROMPTS = (("P1", "Python merge"), ("P2", "mmap vs read"), ("P3", "Bash watcher"))
COLORS = {"GPU":"#0072B2", "ACO":"#D55E00", "CPU":"#009E73",
          "CPU + GPU MTP":"#009E73", "Master CPU + MTP":"#CC79A7",
          "MTP on":"#D55E00", "MTP off":"#0072B2"}
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":12, "axes.titlesize":14,
    "axes.labelsize":12,"xtick.labelsize":10.5,"ytick.labelsize":10.5,"legend.fontsize":12,
    "figure.facecolor":"white","axes.facecolor":"#FCFDFE","axes.edgecolor":"#B8C2CC",
    "axes.labelcolor":"#273444","xtick.color":"#465363","ytick.color":"#465363",
    "grid.color":"#DCE3EA","grid.linewidth":.75,"savefig.facecolor":"white"})


def load_data():
    path=RESEARCH/"chart-data.csv"
    with path.open(newline="",encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    required={"kind","figure","mtp","series","prompt","context_tokens","context_k","rate_tok_s"}
    if not required.issubset(rows[0]): raise ValueError(f"Missing fields: {required-set(rows[0])}")
    for r in rows:
        for k in ("context_tokens","context_k","rate_tok_s"):
            r[k]=float(r[k])
        if not all(math.isfinite(r[k]) for k in ("context_tokens","context_k","rate_tok_s")):
            raise ValueError(f"Nonfinite point: {r}")
    return rows


def style(ax,ylabel):
    ax.set_ylabel(ylabel); ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f"{y:,.0f}"))
    ax.grid(True,color="#DCE3EA",alpha=.9,zorder=0)
    for side in ("top","right"): ax.spines[side].set_visible(False)
    for side in ("left","bottom"): ax.spines[side].set_color("#B8C2CC")
    ax.tick_params(axis="both",length=0,pad=6)


def xcontext(ax,xmax=272):
    ax.set_xlim(0,xmax); ax.xaxis.set_major_locator(MultipleLocator(32))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:.0f}"))
    ax.set_xlabel("Context (K tokens; 1K = 1,024 tokens)")


def artists(ax,points,order):
    result=[]
    for series in order:
        data=sorted((r for r in points if r["series"]==series),key=lambda r:r["context_tokens"])
        if not data: continue
        xs=[r["context_k"] for r in data]; ys=[r["rate_tok_s"] for r in data]
        line,=ax.plot(xs,ys,color=COLORS[series],linewidth=3.5,marker="o",markersize=8.2,
            markerfacecolor=COLORS[series],markeredgecolor="white",markeredgewidth=1,
            solid_capstyle="round",solid_joinstyle="round",label=series,zorder=3)
        result.append((line,xs,ys,series))
    return result


def labels(ax,sets,integer=False,fontsize=8.5):
    ax.figure.canvas.draw(); renderer=ax.figure.canvas.get_renderer(); bounds=ax.get_window_extent(renderer); used=[]
    offsets={"GPU":[(0,11),(0,-17),(12,11),(-12,11),(12,-17),(-12,-17),(0,25),(0,-31),(22,10),(-22,10)],
      "ACO":[(0,-17),(0,11),(12,-17),(-12,-17),(12,11),(-12,11),(0,-31),(0,25),(22,-17),(-22,-17)],
      "CPU":[(0,11),(0,-17),(12,11),(-12,11),(12,-17),(-12,-17),(0,25),(0,-31),(22,10),(-22,10)],
      "CPU + GPU MTP":[(0,11),(0,-17),(12,11),(-12,11),(12,-17),(-12,-17),(0,25),(0,-31),(22,10),(-22,10)],
      "Master CPU + MTP":[(0,-17),(0,11),(12,-17),(-12,-17),(12,11),(-12,11),(0,-31),(0,25),(22,-17),(-22,-17)],
      "MTP on":[(0,11),(0,-17),(12,11),(-12,11),(12,-17),(-12,-17),(0,25),(0,-31)],
      "MTP off":[(0,-17),(0,11),(12,-17),(-12,-17),(12,11),(-12,11),(0,-31),(0,25)]}
    for line,xs,ys,series in sets:
        color=to_hex(tuple(c*.72 for c in to_rgb(line.get_color())))
        for x,y in zip(xs,ys):
            txt=f"{int(round(y)):,}" if integer else f"{y:.1f}"
            t=ax.annotate(txt,(x,y),xytext=(0,11),textcoords="offset points",ha="center",va="bottom",
                fontsize=fontsize,color=color,zorder=5,bbox={"boxstyle":"round,pad=.12","facecolor":"white","edgecolor":"none","alpha":.94},
                arrowprops={"arrowstyle":"-","color":line.get_color(),"linewidth":.7,"alpha":.8,"shrinkA":2,"shrinkB":5})
            chosen=None
            for dx,dy in offsets[series]:
                t.set_position((dx,dy)); box=t.get_window_extent(renderer).expanded(1.05,1.08)
                inside=box.x0>=bounds.x0+1 and box.x1<=bounds.x1-1 and box.y0>=bounds.y0+1 and box.y1<=bounds.y1-1
                if inside and all(not box.overlaps(prev) for prev in used): used.append(box); chosen=(dx,dy); break
            if chosen is None:
                chosen=offsets[series][0]; t.set_position(chosen); used.append(t.get_window_extent(renderer).expanded(1,1))
            t.arrow_patch.set_visible(math.hypot(*chosen)>14)


def legend(fig,order):
    hs=[Line2D([0],[0],color=COLORS[n],lw=3.5,marker="o",markersize=8.2,markerfacecolor=COLORS[n],markeredgecolor="white",label=n) for n in order]
    fig.legend(handles=hs,labels=list(order),loc="upper center",bbox_to_anchor=(.5,.952),ncol=len(order),frameon=False,handlelength=2.3,columnspacing=2.4,handletextpad=.6)


def save(fig,slug):
    ASSETS.mkdir(parents=True,exist_ok=True)
    fig.savefig(ASSETS/f"{slug}.png",dpi=220,bbox_inches="tight")
    svg=ASSETS/f"{slug}.svg"
    fig.savefig(svg,bbox_inches="tight",metadata={"Title":slug.replace("_"," ").title()})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)


def line_panel(ax,points,order,ylabel="Decode speed (tokens/s)",xmax=272,ylim_factor=1.28,integer=False):
    a=artists(ax,points,order); style(ax,ylabel); xcontext(ax,xmax)
    ax.set_ylim(0,max(r["rate_tok_s"] for r in points)*ylim_factor)
    labels(ax,a,integer=integer,fontsize=8 if integer else 9)


def make_triptych(points,slug,state,order=("GPU","ACO","CPU"),series_filter=None,ylabel="Decode speed (tokens/s)",integer=False):
    fig,axes=plt.subplots(3,1,figsize=(18.5,14),sharex=True)
    fig.suptitle(f"{'Prefill' if integer else 'Decode'} speed · MTP {state}" if series_filter is None else "Decode speed · CPU placement · max · reasoning OFF",
                 fontsize=20,fontweight="semibold",color="#1E2B39",y=.99)
    for ax,(prompt,title) in zip(axes,PROMPTS):
        panel=[r for r in points if r["prompt"]==prompt and (series_filter is None or r["series"] in order)]
        if series_filter is None: panel=[r for r in panel if r["mtp"]==state]
        line_panel(ax,panel,order,ylabel,integer=integer)
        ax.set_title(f"{prompt} · {title}",loc="left",fontweight="semibold",fontsize=14,color="#334155",pad=10)
    legend(fig,order); fig.subplots_adjust(left=.105,right=.985,top=.91,bottom=.075,hspace=.34); save(fig,slug)


def make_single(points,slug,title,order,ylabel="Decode speed (tokens/s)",integer=False):
    fig,ax=plt.subplots(figsize=(18.5,9.5)); fig.suptitle(title,fontsize=20,fontweight="semibold",color="#1E2B39",y=.99)
    line_panel(ax,points,order,ylabel=ylabel,integer=integer,ylim_factor=1.16 if integer else 1.28); legend(fig,order); fig.subplots_adjust(left=.105,right=.985,top=.87,bottom=.12); save(fig,slug)


def make_bar(slug,title,subtitle,points,labels_text,colors=None,integer_values=False):
    colors=colors or [COLORS[s] for s in labels_text]
    fig,ax=plt.subplots(figsize=(13.5,7.0)); fig.suptitle(title,fontsize=20,fontweight="semibold",color="#1E2B39",y=.97)
    if subtitle: ax.set_title(subtitle,loc="left",fontsize=12,color="#465363",pad=16)
    vals=[r["rate_tok_s"] for r in points]; y=list(range(len(vals)))
    bars=ax.barh(y,vals,color=colors,height=.58,edgecolor="white",linewidth=1,zorder=3)
    limit=max(vals)*1.32
    for bar,v in zip(bars,vals): ax.text(v+limit*.018,bar.get_y()+bar.get_height()/2,f"{v:.0f} tok/s" if integer_values else f"{v:.2f} tok/s",va="center",fontsize=12,color="#23313F",fontweight="semibold")
    ax.set_yticks(y,labels_text,fontsize=12); ax.invert_yaxis(); ax.set_xlim(0,limit); ax.set_xlabel("Speed (tokens/s)")
    ax.xaxis.grid(True,color="#DCE3EA",alpha=.9,zorder=0); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.spines["left"].set_visible(False); ax.tick_params(axis="y",length=0); ax.tick_params(axis="x",length=0,pad=6)
    fig.subplots_adjust(left=.29,right=.96,top=.84,bottom=.13); save(fig,slug)


def make_long_context(decode):
    fig,axes=plt.subplots(1,2,figsize=(18,8),sharey=True)
    common_max=max(r["rate_tok_s"] for r in decode if r["context_tokens"]==256000 and r["series"] in ("ACO","CPU"))*1.28
    fig.suptitle("Long-context maximum decode",fontsize=20,fontweight="semibold",color="#1E2B39",y=.97)
    for ax,state in zip(axes,("on","off")):
        sub=[r for r in decode if r["mtp"]==state and r["context_tokens"]==256000 and r["series"] in ("ACO","CPU")]
        prompts=["P1","P2","P3"]; xpos=range(3); width=.34
        for i,series in enumerate(("ACO","CPU")):
            vals=[next(r["rate_tok_s"] for r in sub if r["prompt"]==p and r["series"]==series) for p in prompts]
            positions=[x+(i-.5)*width for x in xpos]
            bars=ax.bar(positions,vals,width,color=COLORS[series],label=series,edgecolor="white",linewidth=1,zorder=3)
            for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,v+max(vals)*.025,f"{v:.1f}",ha="center",va="bottom",fontsize=10,color=to_hex(tuple(c*.72 for c in to_rgb(COLORS[series]))),fontweight="semibold")
        ax.set_title(f"MTP {state}",fontweight="semibold",color="#334155"); ax.set_xticks(list(xpos),prompts); ax.set_xlabel("Prompt")
        ax.grid(axis="y",color="#DCE3EA",alpha=.9,zorder=0); ax.set_axisbelow(True); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.set_ylim(0, common_max)
    axes[0].set_ylabel("Decode speed (tokens/s)"); legend(fig,("ACO","CPU")); fig.subplots_adjust(left=.09,right=.98,top=.85,bottom=.13,wspace=.12); save(fig,"17_long_context_decode_comparison")


def main():
    rows=load_data(); decode=[r for r in rows if r["kind"]=="decode"]; cpu=[r for r in rows if r["kind"]=="cpu-placement"]; prefill=[r for r in rows if r["kind"]=="prefill"]
    # 01: per-series selected MTP-on P1 64K-class observations, contexts preserved.
    p=[next(r for r in decode if r["mtp"]=="on" and r["prompt"]=="P1" and r["series"]==s and r["context_tokens"]==c) for s,c in (("GPU",61440),("ACO",62464),("CPU",61440))]
    make_bar("01_mtp_64k_comparison","Maximum decode with MTP · 64K context · Python merge","",p,["GPU","ACO","CPU"])
    make_triptych(decode,"02_mtp_context_comparison","on")
    for i,(prompt,title) in enumerate(PROMPTS,3):
        make_single([r for r in decode if r["mtp"]=="on" and r["prompt"]==prompt],f"{i:02d}_mtp_prompt_{i-2}",f"MTP on · {prompt} · {title}",("GPU","ACO","CPU"))
    p=[next(r for r in decode if r["mtp"]=="off" and r["prompt"]=="P1" and r["series"]==s and r["context_tokens"]==62464) for s in ("GPU","ACO","CPU")]
    make_bar("06_non_mtp_64k_comparison","Maximum decode without MTP · 64K context · Python merge","",p,["GPU","ACO","CPU"])
    make_triptych(decode,"07_non_mtp_context_comparison","off")
    fig,axes=plt.subplots(3,1,figsize=(18.5,14),sharex=True); fig.suptitle("ACO decode · MTP on vs off",fontsize=20,fontweight="semibold",color="#1E2B39",y=.99)
    for ax,(prompt,title) in zip(axes,PROMPTS):
        panel=[dict(r,series="MTP on" if r["mtp"]=="on" else "MTP off") for r in decode if r["series"]=="ACO" and r["prompt"]==prompt]
        line_panel(ax,panel,("MTP on","MTP off")); ax.set_title(f"{prompt} · {title}",loc="left",fontweight="semibold",fontsize=14,color="#334155",pad=10)
    legend(fig,("MTP on","MTP off")); fig.subplots_adjust(left=.105,right=.985,top=.91,bottom=.075,hspace=.34); save(fig,"08_aco_mtp_on_off")
    for i,(prompt,title) in enumerate(PROMPTS,9):
        make_single([r for r in decode if r["mtp"]=="off" and r["prompt"]==prompt],f"{i:02d}_non_mtp_prompt_{i-8}",f"MTP off · {prompt} · {title}",("GPU","ACO","CPU"))
    make_triptych(cpu,"12_cpu_mtp_placement","on",("CPU + GPU MTP","Master CPU + MTP"),series_filter=True)
    p=[next(r for r in prefill if r["mtp"]=="on" and r["series"]==s and r["context_tokens"]==32769) for s in ("GPU","ACO")]
    make_bar("13_prefill_mtp_on_32k_comparison","Prefill with MTP · 32K context","",p,["GPU","ACO"],integer_values=True)
    make_single([r for r in prefill if r["mtp"]=="on"],"14_prefill_mtp_on_context","Prefill speed · MTP on",("GPU","ACO","CPU"),ylabel="Prefill speed (tokens/s)",integer=True)
    p=[next(r for r in prefill if r["mtp"]=="off" and r["series"]==s and r["context_tokens"]==c) for s,c in (("GPU",130049),("ACO",131073),("CPU",128031))]
    make_bar("15_prefill_mtp_off_128k_comparison","Prefill without MTP · 128K context","",p,["GPU","ACO","CPU"], [COLORS[s] for s in ("GPU","ACO","CPU")],integer_values=True)
    make_single([r for r in prefill if r["mtp"]=="off"],"16_prefill_mtp_off_context","Prefill speed · MTP off",("GPU","ACO","CPU"),ylabel="Prefill speed (tokens/s)",integer=True)
    make_long_context(decode)
    captions=[
      ("01_mtp_64k_comparison","MTP on at 64K-class context","On the P1 Python prompt, ACO measured 79.69 tokens/s: 4.5× CPU at 17.68 and 99% of GPU at 80.70."),
      ("02_mtp_context_comparison","MTP-on decode across context lengths","Near 250K context, ACO measured 61.13, 48.83, and 55.35 tokens/s on P1–P3, versus CPU at 7.21, 5.49, and 6.50: 8.47–8.89× CPU speed."),
      ("03_mtp_prompt_1","MTP on · P1 Python merge","On P1, ACO measured 102.48 tokens/s at 6K, 70.61 at 128K, and 61.23 near 256K."),
      ("04_mtp_prompt_2","MTP on · P2 mmap vs read","Near 250K on P2, ACO measured 48.83 tokens/s versus CPU at 5.49: 8.89× CPU speed."),
      ("05_mtp_prompt_3","MTP on · P3 Bash watcher","Near 250K on P3, ACO measured 55.35 tokens/s versus CPU at 6.50: 8.51× CPU speed."),
      ("06_non_mtp_64k_comparison","MTP off at 62,464 tokens","On P1, ACO measured 42.33 tokens/s versus CPU at 8.48 and GPU at 43.01: 5.0× CPU speed and 98% of GPU speed."),
      ("07_non_mtp_context_comparison","MTP-off decode across context lengths","Near 256K, ACO measured about 37.5 tokens/s across P1–P3, versus CPU at about 3.0 tokens/s."),
      ("08_aco_mtp_on_off","ACO decode · MTP on versus off","On P1 near 256K, ACO measured 61.23 tokens/s with MTP on and 37.59 with MTP off."),
      ("09_non_mtp_prompt_1","MTP off · P1 Python merge","Near 250K on P1, ACO measured 37.51 tokens/s versus CPU at 2.98: 12.6× CPU speed."),
      ("10_non_mtp_prompt_2","MTP off · P2 mmap vs read","Near 250K on P2, ACO measured 37.57 tokens/s versus CPU at 2.98: 12.6× CPU speed."),
      ("11_non_mtp_prompt_3","MTP off · P3 Bash watcher","Near 250K on P3, ACO measured 37.45 tokens/s versus CPU at 2.98: 12.6× CPU speed."),
      ("12_cpu_mtp_placement","CPU MTP placement","Around 64K on P1–P3, CPU + GPU MTP measured 17.68, 13.56, and 15.29 tokens/s; master CPU + MTP measured 15.12, 12.26, and 13.84. Around 128K, CPU + GPU MTP measured 14–19% higher."),
      ("13_prefill_mtp_on_32k_comparison","MTP-on prefill at 32K target","At 32K, GPU measured 924 and ACO 832 fresh tokens/s; ACO reached 90% of the GPU rate."),
      ("14_prefill_mtp_on_context","MTP-on prefill across context lengths","ACO measured 1,402 fresh tokens/s on the first append and 614 near 256K. Around 64K, ACO measured 817 versus GPU at 559."),
      ("15_prefill_mtp_off_128k_comparison","Prefill without MTP · 128K context","Near 128K, ACO measured 850 fresh tokens/s, CPU 559, and GPU 318; ACO reached 2.7× the GPU rate."),
      ("16_prefill_mtp_off_context","MTP-off prefill across context lengths","ACO measured 1,589 fresh tokens/s on the first append and 803 near 256K."),
      ("17_long_context_decode_comparison","Long-context maximum decode","At 250K filled context, ACO measured 8.47–8.89× CPU speed with MTP on and 12.55–12.59× with MTP off across P1–P3."),
    ]
    (RESEARCH/"chart_captions.json").write_text(json.dumps([{"file":f,"title":t,"caption":c} for f,t,c in captions],indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Rendered 17 chart pairs from {len(rows)} curated observations")

if __name__=="__main__": main()
