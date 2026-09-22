"""Draw the SAFE-EDA architecture with the manuscript's original geometry."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

from .plotting import format_figure

BLUE, ORANGE, TEAL = '#236B94', '#BE663E', '#278878'
INK, GRAY, LIGHT = '#243846', '#647580', '#DCE4E8'
PALE_BLUE, PALE_ORANGE, PALE_TEAL = '#EDF4F8', '#FCF1EA', '#EDF6F2'
STYLE = {'font.family':'DejaVu Sans', 'font.size':7.2,
         'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
         'axes.linewidth':.5, 'text.color':INK}


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, w), ylim=(0, h))
    ax.set_aspect('equal'); ax.axis('off')
    return fig, ax

def text(ax, x, y, s, fs=7.2, c=INK, weight='normal', ha='left', **kw):
    return ax.text(x, y, s, fontsize=fs, color=c, weight=weight,
                   ha=ha, va='center', **kw)

def line(ax, points, c=GRAY, lw=.7, ls='-', **kw):
    a=np.asarray(points); ax.plot(a[:,0],a[:,1], color=c, lw=lw, ls=ls, **kw)

def arrow(ax, points, c=GRAY, lw=.8, ls='-'):
    if len(points)>2: line(ax, points[:-1], c, lw, ls)
    ax.annotate('', xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle='-|>', color=c, lw=lw,
                                linestyle=ls, mutation_scale=7, shrinkA=0, shrinkB=0))

def rect(ax, x,y,w,h,fc='white',ec=LIGHT,lw=.6,r=.035):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                     fc=fc,ec=ec,lw=lw)
    ax.add_patch(p);return p

def node(ax,x,y,w,h,label,fc='white',c=INK,fs=7.1,ec=LIGHT):
    rect(ax,x,y,w,h,fc,ec);text(ax,x+w/2,y+h/2,label,fs,c,ha='center')

def header(ax,x,y,letter,title,w,c=INK):
    text(ax,x,y,letter,10,c,'bold');text(ax,x+.22,y,title,8.6,c,'bold')
    line(ax,[(x,y-.17),(x+w,y-.17)],LIGHT,.65)

def plus(ax,x,y):
    ax.add_patch(Circle((x,y),.075,fc='white',ec=GRAY,lw=.7));text(ax,x,y,'+',8,GRAY,ha='center')

def _architecture(output: Path) -> Path:
    fig, ax=canvas(7.16,3.50)
    header(ax,.05,3.38,'a','Target model & transfer boundary',3.13)
    header(ax,3.40,3.38,'b','Inside each residual block',3.71)
    line(ax,[(3.29,.10),(3.29,3.18)],LIGHT,.65)
    # a. Trainable trunk and deterministic side path are explicitly separate.
    text(ax,.20,3.02,'x, Δx, Δ²x, 1',7.6,INK,'bold')
    text(ax,1.54,3.02,'60-s input',7.1,GRAY)
    rect(ax,.11,1.90,1.47,.91,PALE_BLUE,'none')
    text(ax,.22,2.68,'TRANSFERRED TRUNK',7.0,BLUE,'bold')
    node(ax,.23,2.24,1.22,.31,'Stem k5 · 4 → 64','white',BLUE,7.2,ec='#C2D9E7')
    arrow(ax,[(.82,2.91),(.82,2.81)],BLUE)
    # Continue arrow through the trunk title only to a clear edge.
    arrow(ax,[(.82,2.62),(.82,2.55)],BLUE)
    for dx,dy in [(.035,.045),(0,0)]:rect(ax,.23+dx,1.98+dy,1.22,.18,'white','#C2D9E7')
    text(ax,.84,2.07,'6 × residual block',7.2,BLUE,ha='center')
    arrow(ax,[(.82,2.24),(.82,2.205)],BLUE)
    # x alone feeds the fixed decomposition; task gradients do not enter this branch.
    node(ax,1.91,2.39,1.22,.43,'Fixed\ndecomposition',PALE_TEAL,TEAL,7.2,ec='none')
    arrow(ax,[(.98,2.90),(2.52,2.90),(2.52,2.82)],TEAL)
    text(ax,1.69,2.82,'x',7.0,TEAL)
    text(ax,2.52,2.22,'No task gradients',6.9,TEAL,ha='center')
    arrow(ax,[(1.91,2.51),(1.74,2.51),(1.74,2.075),(1.50,2.075)],TEAL,.7,'--')
    text(ax,1.70,1.91,'Gate',6.6,TEAL,ha='center')
    text(ax,1.03,1.73,'296,822 parameters',7.0,BLUE)
    # Feature bus to global and event pooling.
    arrow(ax,[(.82,1.97),(.82,1.56),(.50,1.56),(.50,1.40)],BLUE)
    arrow(ax,[(.82,1.56),(1.58,1.56),(1.58,1.40)],BLUE)
    # Detached information supplies event weights and the physiological summary.
    line(ax,[(3.13,2.60),(3.18,2.60),(3.18,.58)],TEAL,.7,'--')
    arrow(ax,[(3.18,1.56),(2.78,1.56),(2.78,1.40)],TEAL,.7,'--')
    arrow(ax,[(2.78,1.56),(1.90,1.56),(1.90,1.40)],TEAL,.7,'--')
    arrow(ax,[(3.18,.58),(2.41,.58)],TEAL,.7,'--')
    text(ax,2.70,.42,'Scales',6.7,TEAL)
    pools=[(.08,.88,'Global','128 → 64 → 64','12,416'),(1.15,.91,'Event','256 → 128 → 64','41,152'),(2.23,.91,'Physio','10 → 64 → 64','4,864')]
    for x,w,label,dims,n in pools:
        rect(ax,x,.93,w,.47,PALE_ORANGE,'none')
        text(ax,x+w/2,1.27,label,7.6,ORANGE,'bold',ha='center')
        text(ax,x+w/2,1.08,dims,6.6,ORANGE,ha='center')
        text(ax,x+w/2+.09,.81,n,6.7,GRAY,ha='left')
    for x in [.52,1.60,2.685]:
        line(ax,[(x,.94),(x,.76),(1.59,.76)],ORANGE,.65)
    arrow(ax,[(1.59,.76),(1.59,.74)],ORANGE,.65)
    node(ax,.78,.43,1.63,.31,'Scaled sum + LayerNorm',PALE_ORANGE,ORANGE,7.2,ec='none')
    arrow(ax,[(1.59,.43),(1.59,.33)],ORANGE)
    node(ax,.78,.035,1.63,.28,'Classifier · 64 → 64 → 3',PALE_ORANGE,ORANGE,7,ec='none')
    text(ax,.12,.56,'Fresh',6.8,ORANGE,'bold')
    text(ax,.12,.38,'62,917',7.2,ORANGE,'bold')
    # b. Four depthwise experts, explicit parallel branch and residual routes.
    text(ax,3.45,3.035,'h',8.0,BLUE,'bold')
    arrow(ax,[(3.59,3.035),(3.79,3.035)],BLUE,.65)
    node(ax,3.79,2.91,.86,.25,'GroupNorm',PALE_BLUE,BLUE,7,ec='none')
    arrow(ax,[(4.65,3.035),(4.79,3.035),(4.79,2.86)],BLUE,.65)
    text(ax,4.96,3.035,'Depthwise experts · k7',7.3,BLUE,'bold')
    centers=[3.86,4.62,5.38,6.14]
    line(ax,[(3.52,2.86),(6.67,2.86)],BLUE,.7)
    for x,k in zip(centers,range(1,5)):
        arrow(ax,[(x,2.86),(x,2.76)],BLUE,.65)
        node(ax,x-.31,2.42,.62,.34,f'Expert {k}',PALE_BLUE,BLUE,7.3,ec='none')
        # Filter taps are a structure glyph, not measured activation values.
        for dx in np.linspace(-.20,.20,7):ax.add_patch(Circle((x+dx,2.47),.012,fc=BLUE,ec='none'))
        line(ax,[(x,2.42),(x,2.32)],BLUE,.65)
    line(ax,[(3.86,2.32),(6.14,2.32)],BLUE,.7)
    arrow(ax,[(5.72,2.32),(5.72,2.19)],BLUE)
    node(ax,3.46,1.91,1.30,.29,'Gate · 9 → 32 → 4',PALE_TEAL,TEAL,7.1,ec='none')
    text(ax,4.11,1.79,'Physiological features',6.7,TEAL,ha='center')
    node(ax,4.99,1.83,1.50,.36,'Static + gated mixture\n' + r'$s + \beta c(g-s)$',PALE_BLUE,BLUE,7.1,ec='none')
    arrow(ax,[(4.76,2.055),(4.99,2.055)],TEAL,.7,'--')
    text(ax,4.11,1.54,'Learned static weights',6.7,GRAY,ha='center')
    arrow(ax,[(4.77,1.55),(4.88,1.55),(4.88,1.97),(4.99,1.97)],GRAY,.65)
    arrow(ax,[(5.72,1.83),(5.72,1.78)],BLUE)
    node(ax,4.99,1.49,1.50,.29,'1×1 mix → GLU → 1×1',PALE_BLUE,BLUE,7.0,ec='none')
    plus(ax,6.71,1.63);arrow(ax,[(6.49,1.63),(6.635,1.63)],BLUE,.65)
    line(ax,[(3.67,3.035),(3.67,3.17),(7.02,3.17),(7.02,1.63)],GRAY,.65)
    arrow(ax,[(7.02,1.63),(6.785,1.63)],GRAY,.65)
    text(ax,6.98,2.31,'h',7.1,GRAY,ha='right')
    arrow(ax,[(6.71,1.555),(6.71,1.40),(5.72,1.40),(5.72,1.32)],BLUE,.65)
    node(ax,4.99,1.03,1.50,.29,'GN + FFN · 64 → 256 → 64',PALE_BLUE,BLUE,6.7,ec='none')
    plus(ax,6.71,1.17);arrow(ax,[(6.49,1.17),(6.635,1.17)],BLUE,.65)
    arrow(ax,[(6.71,1.40),(7.02,1.40),(7.02,1.17),(6.785,1.17)],GRAY,.65)
    arrow(ax,[(6.71,1.095),(6.71,.94)],BLUE,.65)
    text(ax,4.11,1.20,'49,225 parameters',7,BLUE,'bold',ha='center')
    # c. Per-expert duration discretization; never presented as full-network RF.
    line(ax,[(3.41,.87),(7.10,.87)],LIGHT,.6)
    text(ax,3.42,.72,'c',9,INK,'bold');text(ax,3.63,.72,'Expert scales across sampling rates',7.7,INK,'bold')
    for x,v in zip([4.98,5.57,6.16,6.75],['.5','1','2','4']):text(ax,x,.52,v,7.1,INK,ha='center')
    text(ax,3.47,.52,'Nominal duration (s)',6.8,GRAY)
    text(ax,3.47,.32,'Dilation: 32 → 4 Hz',6.8,GRAY)
    for x,v in zip([4.98,5.57,6.16,6.75],['2 → 1','5 → 1','10 → 1','21 → 2']):text(ax,x,.32,v,7,BLUE,ha='center')
    text(ax,3.47,.12,'Target support (s)',6.8,GRAY)
    for x,v in zip([4.98,5.57,6.16,6.75],['1.75','1.75','1.75','3.25']):text(ax,x,.12,v,7,INK,ha='center')
    format_figure(fig)
    path = output / 'fig2_architecture_detail.pdf'
    fig.savefig(path, dpi=400, facecolor='white')
    plt.close(fig)
    return path

def architecture(output: Path) -> Path:
    """Write the architecture PDF without loading waveform data or image assets."""
    with plt.style.context('default'), plt.rc_context(STYLE):
        return _architecture(output)

