"""Reproduce the manuscript figures that use summary tables or model structure."""
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

BLUE, ORANGE = '#0072B2', '#D55E00'
HOPS = [1, 4, 60, 240]
MAIN_STYLE = {'font.family':'DejaVu Sans', 'font.size':7.5,
    'axes.labelsize':7.5, 'axes.titlesize':8, 'xtick.labelsize':7,
    'ytick.labelsize':7, 'legend.fontsize':7, 'pdf.fonttype':42,
    'axes.spines.top':False, 'axes.spines.right':False,
    'axes.linewidth':.6, 'lines.linewidth':1.2, 'lines.markersize':3.8}
DETAIL_STYLE = {'font.family':'DejaVu Sans','font.size':7.5,'axes.labelsize':7.5,
    'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
    'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.6,'lines.linewidth':1.0}


def leading_zero(text):
    return re.sub(r'(?<![\w.])\.(?=\d)', '0.', str(text))

def format_figure(fig):
    fig.canvas.draw()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            original = axis.get_major_formatter()
            axis.set_major_formatter(FuncFormatter(
                lambda value, pos, original=original:
                leading_zero(original(value, pos))))
    for label in fig.findobj(Text):
        label.set_text(leading_zero(label.get_text()))

def save(fig, name: str, output: Path) -> Path:
    format_figure(fig)
    # Preserve the physical dimensions; a tight bounding box changes the scale.
    path = output / f'{name}.pdf'
    fig.savefig(path)
    plt.close(fig)
    return path


def panel(ax, letter, title):
    ax.set_title(f'{letter}  {title}', loc='left', pad=9, fontweight='bold')

def atlas(tables: Path, output: Path) -> Path:
    DATA = tables
    BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#69747B"
    t05=pd.read_csv(DATA/'T05_normalization_ladder.csv')
    t06=pd.read_csv(DATA/'T06_hop_ladder.csv')
    t17=pd.read_csv(DATA/'T17_normalization_hop_cross_grid.csv')
    fig=plt.figure(figsize=(7.16,2.70))
    a=fig.add_axes([.175,.655,.32,.245]); b=fig.add_axes([.645,.655,.335,.245])
    c=fig.add_axes([.175,.145,.32,.28]); d=fig.add_axes([.645,.145,.335,.28])
    q=t17.pivot(index='normalization',columns='hop_samples',values='delta').loc[['global_train','per_subject_z'],HOPS]
    cmap=LinearSegmentedColormap.from_list('gain',['#F4F7F8','#ACD4E6','#0072B2'])
    a.imshow(q.to_numpy(),cmap=cmap,vmin=0,vmax=.24,aspect='auto')
    a.set_xticks(range(4),HOPS); a.set_yticks([0,1],['Global-train','Per-subject-z'])
    for i in range(2):
        for j in range(4):
            v=q.iloc[i,j]; a.text(j,i,f'{v:+.3f}',ha='center',va='center',color='white' if v>.15 else '#1B303D',fontsize=8)
    a.set_xlabel('Hop (samples)'); panel(a,'a','WESAD gain (Δ macro-F1)')
    for norm,color,label in [('global_train',ORANGE,'Global'),('per_subject_z',GREEN,'Subject')]:
        q=t17[t17.normalization==norm].set_index('hop_samples').loc[HOPS]
        b.plot(range(4),q.scratch_macro_f1,'o--',color=color,label=f'{label} scratch',mfc='white')
        b.plot(range(4),q.transfer_macro_f1,'s-',color=color,label=f'{label} transfer')
    b.set_xticks(range(4),HOPS); b.set_xlim(-.24,3.24); b.set_ylim(0,.77)
    b.set_xlabel('Hop (samples)'); b.set_ylabel('Macro-F1',labelpad=2)
    panel(b,'b','WESAD all arms'); b.set_yticks([0,.4,.6]); b.grid(axis='y',alpha=.18)
    b.legend(loc='lower center',bbox_to_anchor=(.5,.01),ncol=2,frameon=False,
             handlelength=1.5,columnspacing=.8,handletextpad=.4,fontsize=6.6,
             borderaxespad=0,borderpad=0,labelspacing=.15)
    # Reset indices: data positions and tick positions share this exact vector.
    q=t05[t05.dataset=='wearable'].iloc[::-1].reset_index(drop=True)
    y=np.arange(len(q)); ci=np.array([[float(n) for n in v.strip('[]').split(',')] for v in q.ci95])
    c.hlines(y,ci[:,0],ci[:,1],color=BLUE,lw=1.2)
    c.scatter(q.delta,y,color=BLUE,s=16,zorder=3)
    c.set_yticks(y,q.normalization.str.replace('_','-'))
    c.set_ylim(4.6,-.75); c.set_xlim(-.04,.18)
    c.axvline(0,color=GRAY,lw=.6); c.set_xticks([-.02,.02,.06,.10])
    c.set_xlabel('Transfer − scratch macro-F1')
    for yi,r in zip(y,q.itertuples()):
        c.text(.112,yi,f'{r.wins}  {r.cohens_dz:.3f}',fontsize=6.8,va='center')
    c.text(.112,-.6,'Wins    dᵤ',fontsize=6.8,va='center')
    # Use proper Cohen's dz glyph via mathtext, with the same legible size.
    c.texts[-1].set_text(r'Wins     $d_z$')
    panel(c,'c','Wearable normalization (n = 26)')
    q=t06[t06.dataset=='wearable'].sort_values('hop_samples').reset_index(drop=True)
    d.plot(range(3),q.delta,'^-',color=BLUE)
    d.set_xticks(range(3),q.hop_samples); d.set_xlabel('Hop (samples)')
    d.set_ylabel('Δ macro-F1'); d.set_xlim(-.23,2.40); d.set_ylim(-.006,.119)
    d.axhline(0,color=GRAY,lw=.6,ls='--'); d.grid(axis='y',alpha=.18)
    for i,r in q.iterrows():
        offset=(-5,-16) if i==2 else (0,10)
        d.annotate(f'{r.wins}',(i,r.delta),xytext=offset,textcoords='offset points',ha='center',fontsize=7)
    panel(d,'d','Wearable hop ladder (n = 26)')
    return save(fig, 'fig4_experiment_atlas', output)

def subjects(tables: Path, output: Path) -> Path:
 DATA = tables
 d=pd.read_csv(DATA/'T21_subject_level_grid.csv');q=d.pivot(index='held_out_subject',columns=['normalization','hop_samples','arm'],values='macro_f1');ids=sorted(q.index,key=lambda s:int(s[1:]));cols=[(n,h) for n in ['global_train','per_subject_z'] for h in [1,4,60,240]]
 a=np.column_stack([(q[(n,h,'safe_edabe_transfer')]-q[(n,h,'safe_scratch')]).loc[ids] for n,h in cols]);assert a.shape==(15,8)
 fig=plt.figure(figsize=(3.5,3.30));ax=fig.add_axes([.14,.195,.81,.69]);cmap=LinearSegmentedColormap.from_list('effect',[ORANGE,'#FAFAF9',BLUE]);lim=max(abs(a.min()),abs(a.max()));im=ax.imshow(a,cmap=cmap,norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto');ax.set_yticks(range(15),ids);ax.set_xticks(range(8),[1,4,60,240]*2);ax.tick_params(length=0);ax.axvline(3.5,color='white',lw=3)
 ax.text(1.5,-1.25,'Global-train',ha='center',fontsize=8,fontweight='bold');ax.text(5.5,-1.25,'Per-subject-z',ha='center',fontsize=8,fontweight='bold');fig.text(.545,.982,'Hop (samples)',ha='center',va='top',fontsize=7);ax.set_ylabel('Held-out subject');
 ax.set_xticks(np.arange(-.5,8),minor=True);ax.set_yticks(np.arange(-.5,15),minor=True);ax.grid(which='minor',color='white',lw=.4);ax.tick_params(which='minor',length=0)
 ca=fig.add_axes([.29,.105,.54,.033]);cb=fig.colorbar(im,cax=ca,orientation='horizontal');cb.set_label('Transfer − scratch macro-F1',labelpad=2);cb.ax.tick_params(length=2)
 return save(fig, 'fig5_subject_effects', output)

def supervision(tables: Path, output: Path) -> Path:
 DATA = tables
 GRAY = "#687A86"
 fig,ax=plt.subplots(figsize=(3.5,2.18));fig.subplots_adjust(left=.33,right=.97,bottom=.25,top=.87)
 ladder=pd.read_csv(DATA/'T14_method_ladder.csv').set_index('arm').macro_f1
 permutation=pd.read_csv(DATA/'T08_label_permutation.csv').set_index('contrast').delta
 scratch=float(ladder['safe_scratch_B1'])
 names=['Scratch','Self-supervised','Shuffled labels','True labels'];vals=[scratch,float(ladder['ssl_edabe_transfer']),scratch+float(permutation['shuffled - scratch (pretraining itself)']),float(ladder['safe_edabe_transfer'])];colors=[GRAY,GRAY,ORANGE,BLUE];y=np.arange(4)
 ax.hlines(y,.49,vals,color=colors,lw=2);ax.scatter(vals,y,c=colors,s=25,zorder=3)
 for yy,x in zip(y,vals):ax.annotate(f'{x:.4f}',(x,yy),xytext=(4,4),textcoords='offset points',fontsize=7)
 ax.set_yticks(y,names);ax.invert_yaxis();ax.set_ylim(3.55,-.7);ax.set_xlim(.49,.61);ax.set_xticks([.50,.55,.60]);ax.set_xlabel('Mean macro-F1');ax.set_title('WESAD · global-train · hop 1',fontsize=8,loc='left');ax.grid(axis='x',alpha=.15)
 return save(fig, 'fig6_supervision_ladder', output)

def plot_figures(tables: Path, output: Path) -> list[Path]:
    """Write Figures 2, 4, 5, and 6 as PDFs from the supplied table directory."""
    from .schematics import architecture

    tables, output = Path(tables), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = [architecture(output)]
    with plt.style.context('default'), plt.rc_context(MAIN_STYLE):
        paths.append(atlas(tables, output))
    with plt.style.context('default'), plt.rc_context(DETAIL_STYLE):
        paths.append(subjects(tables, output))
        paths.append(supervision(tables, output))
    return paths
