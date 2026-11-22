"""Fresh publication figures. Run with the workspace .venv Python."""
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from PIL import Image, ImageOps, ImageDraw
import argparse

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
parser = argparse.ArgumentParser()
parser.add_argument('--palette', choices=['original', 'I'], default='I')
args = parser.parse_args()
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
TEAL, BLUE, ORANGE, INK = '#087F8C', '#4263A8', '#D88236', '#253449'
GREY = '#8C99A8'
NEG = '#C66448'
if args.palette == 'I':
    BLUE, TEAL, ORANGE, NEG = '#638CC0', '#8DB5AE', '#CD8196', '#CD8196'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
 'axes.edgecolor': '#C7CFD8', 'axes.spines.top': False, 'axes.spines.right': False,
 'axes.titleweight': 'bold', 'axes.titlesize': 11, 'axes.labelsize': 9,
 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})

def read_xlsx(path):
    with zipfile.ZipFile(path) as z:
        ss = []
        if 'xl/sharedStrings.xml' in z.namelist():
            ss = [''.join(s.itertext()) for s in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS)]
        sheets = ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet', NS)
        result = {}
        for i, sheet in enumerate(sheets, 1):
            cells = {}
            for c in ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml')).findall('.//m:sheetData/m:row/m:c', NS):
                v = c.find('m:v', NS)
                value = v.text if v is not None else None
                if c.attrib.get('t') == 's': value = ss[int(value)]
                elif c.attrib.get('t') == 'inlineStr': value = ''.join(c.find('m:is', NS).itertext())
                elif value is not None:
                    try: value = float(value)
                    except ValueError: pass
                cells[c.attrib['r']] = value
            result[sheet.attrib['name']] = cells
        return result

W = read_xlsx(ROOT / 'paper_results/paper_tables.xlsx')
manifest = {}
def save(fig, name, source):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight', pad_inches=.16)
    manifest[name] = source
    plt.close(fig)

def panel(ax, letter, title):
    ax.set_title(f'{letter}   {title}', loc='left', pad=14)

def grid(ax, axis='x'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color='#E8ECF1', lw=.7)
    ax.tick_params(length=3)

# 1. Benchmark: horizontal grouped bars, ordered consistently across all panels.
t = W['Table1-novobench_performance']
methods = [str(t[f'A{r}']).replace('Ours', 'RocNovo') for r in range(4, 12)]
order = [7, 5, 6, 4, 3, 2, 0, 1]
fig, axs = plt.subplots(1, 3, figsize=(12.8, 5.2), sharey=True)
for ax, cols, letter, title in zip(axs, [('H','J','L'),('I','K','M'),('B','D','F')], 'ABC', ['Peptide recall','PR-AUC','AA precision']):
    for j, (col, color) in enumerate(zip(cols, [BLUE, TEAL, ORANGE])):
        vals = np.array([t[f'{col}{4+i}'] for i in order])*100
        ax.barh(np.arange(8)+(j-1)*.23, vals, height=.20, color=color, label=['Seven-species','Nine-species','HC-PT'][j])
    ax.set_yticks(range(8), [methods[i] for i in order])
    ax.set_xlim(0, 90 if title=='AA precision' else 72)
    ax.set_xlabel('Score (%)'); panel(ax, letter, title); grid(ax)
    ax.axhspan(-.45,.45,color=TEAL,alpha=.07,zorder=0)
axs[0].invert_yaxis()
axs[0].get_yticklabels()[0].set_fontweight('bold')
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.53,.005))
fig.subplots_adjust(left=.13,right=.99,top=.9,bottom=.17,wspace=.13)
save(fig,'fig02_novobench','paper_tables.xlsx / Table1-novobench_performance / A3:M11')

# 2. Per-species gain over each baseline; fixed zero-centred scale across panels.
s = W['Table2-9species-v1']
species = [s[f'{c}1'] for c in 'DEFGHIJKL']
cmp = LinearSegmentedColormap.from_list('gain',[NEG,'#F8F8F5',BLUE if args.palette == 'I' else TEAL])
fig, axs = plt.subplots(2,2,figsize=(12.8,6.1))
for ax, start, letter, title in zip(axs.flat,[2,7,12,17],'ABCD',['V1 · AA precision','V1 · Peptide recall','V2 (filtered) · AA precision','V2 (filtered) · Peptide recall']):
    a = np.array([[s[f'{c}{start+4}']-s[f'{c}{r}'] for c in 'DEFGHIJKL'] for r in range(start,start+4)])*100
    im=ax.imshow(a,cmap=cmp,norm=TwoSlopeNorm(vmin=-16,vcenter=0,vmax=16),aspect='auto')
    ax.set_yticks(range(4),[s[f'C{r}'] for r in range(start,start+4)])
    ax.set_xticks(range(9),species,rotation=35,ha='right',fontsize=8)
    for (i,j),v in np.ndenumerate(a):
        ax.text(j,i,f'{v:+.1f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>10 else INK)
    ax.tick_params(length=0); panel(ax,letter,title)
    for sp in ax.spines.values():sp.set_visible(False)
fig.subplots_adjust(left=.10,right=.90,bottom=.14,top=.92,wspace=.25,hspace=.75)
cax=fig.add_axes([.93,.23,.012,.56]); fig.colorbar(im,cax=cax,label='RocNovo − baseline (percentage points)')
save(fig,'fig03_species_gains','paper_tables.xlsx / Table2-9species-v1 / C1:L21; species-level differences, no averaging')

# 3. Discrete ablations and beam response, separate y scales by metric.
a=W['Table4-AblationStudy']; b=W['Table5-BeamStudy']
fig=plt.figure(figsize=(12.8,4.8)); gs=fig.add_gridspec(2,2,width_ratios=[1.18,1],hspace=.5,wspace=.35)
ax=fig.add_subplot(gs[:,0]); x=np.arange(5)
for col,off,color,label in [('D',-.17,BLUE,'AA precision'),('F',.17,TEAL,'Peptide recall')]:
    vals=[a[f'{col}{r}']*100 for r in range(3,8)]
    bars=ax.bar(x+off,vals,width=.29,color=color,label=label)
    ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
ax.set_xticks(x,['Base','Base\n+ CL','Base\n+ RoPE','Base\n+ CL + RoPE','Full\n+ CL + RoPE + BD'],fontsize=8)
ax.set_ylim(0,95);ax.set_ylabel('Score (%)');grid(ax,'y');panel(ax,'A','Module ablation');ax.legend(frameon=False,loc='upper left',fontsize=8)
beams=[1,2,5,10,15,20]
for idx,row,color,title in [(0,11,BLUE,'AA recall'),(1,21,TEAL,'Peptide recall')]:
    ax=fig.add_subplot(gs[idx,1]); vals=np.array([b[f'{c}{row}'] for c in 'CDEFGH'])*100
    ax.plot(beams,vals,'o-',color=color,lw=2,ms=4)
    ax.axvline(10,color=GREY,ls='--',lw=1)
    ax.annotate(f'{vals[3]:.2f}%',(10,vals[3]),xytext=(8,-16),textcoords='offset points',fontsize=8)
    ax.set_xticks(beams);ax.set_ylabel('Macro recall (%)');grid(ax,'y');panel(ax,'BC'[idx],title)
    if idx:ax.set_xlabel('Beam size')
fig.subplots_adjust(left=.07,right=.98,bottom=.16,top=.88)
fig.text(.07,.015,'Base includes precursor-input and Pre-LN changes. Beam curves: arithmetic mean across nine species on V1.',fontsize=8,color='#647286')
save(fig,'fig04_ablation_beam','paper_tables.xlsx / Table4-AblationStudy / A2:F7; Table5-BeamStudy / C11:H11, C21:H21')

# 4. PTM: single operating points, missing ReNovo values are omitted.
fig,axs=plt.subplots(1,3,figsize=(12,4.5),sharex=True,sharey=True)
markers=['o','s','^','D','v','P','*']; colors=['#A5AFBB','#7C8EA5','#9B88B1','#B6A26C',ORANGE,BLUE,TEAL]
if args.palette == 'I':
    colors=['#B4BBC6','#8F9EB4','#A99CC1','#C3AD99',ORANGE,BLUE,TEAL]
for ax,cols,letter,title in zip(axs,[('B','C'),('D','E'),('F','G')],'ABC',['Seven-species','Nine-species','HC-PT']):
    for i,r in enumerate([16,17,18,19,20,21,23]):
        px,ry=t[f'{cols[0]}{r}']*100,t[f'{cols[1]}{r}']*100
        name=str(t[f'A{r}']).replace('Ours','RocNovo')
        ax.scatter(px,ry,s=140 if r==23 else 65,marker=markers[i],color=colors[i],edgecolor='white',linewidth=.7,label=name,zorder=4 if r==23 else 3)
        if r==23:ax.annotate('RocNovo',(px,ry),xytext=(0,10),textcoords='offset points',ha='center',fontweight='bold',fontsize=8)
    ax.set_xlim(5,90);ax.set_ylim(5,90);ax.set_aspect('equal');ax.set_xlabel('PTM precision (%)');grid(ax,'both');panel(ax,letter,title)
axs[0].set_ylabel('PTM recall (%)')
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=7,frameon=False,fontsize=8,bbox_to_anchor=(.5,.005))
fig.subplots_adjust(left=.07,right=.99,top=.88,bottom=.21,wspace=.15)
save(fig,'figS01_ptm','paper_tables.xlsx / Table1-novobench_performance / A16:G23; ReNovo absent, omitted')

# 5. Dataset overview; species totals describe source overview, not presumed test sizes.
d=read_xlsx(ROOT/'dataset-overview/dataset_overview.xlsx')['Sheet1']
fig,axs=plt.subplots(1,3,figsize=(12.8,4.7),gridspec_kw={'width_ratios':[1,1,1]})
vals=np.array([d[f'C{r}'] for r in [2,3,6,15]])
axs[0].barh(range(4),vals,color=[INK,ORANGE,BLUE,TEAL],height=.55)
axs[0].set_xscale('log');axs[0].set_yticks(range(4),['MassIVE-KB','NovoBench','9-species V1','9-species V2'])
axs[0].invert_yaxis();axs[0].set_xlim(5e5,8e7);axs[0].set_xlabel('PSM count (log scale)');grid(axs[0]);panel(axs[0],'A','Dataset scale')
for i,v in enumerate(vals):axs[0].text(v*1.08,i,f'{v/1e6:.2f}M',va='center',fontsize=8)
for ax,start,color,letter,title in zip(axs[1:],[6,15],[BLUE,TEAL],'BC',['V1 species composition','V2 species composition']):
    counts=np.array([d[f'E{r}'] for r in range(start,start+9)])
    pct=counts/counts.sum()*100
    ax.barh(range(9),pct,color=color,height=.62)
    ax.set_yticks(range(9),species,fontsize=8);ax.invert_yaxis();ax.set_xlim(0,55);ax.set_xlabel('Share of source PSMs (%)');grid(ax);panel(ax,letter,title)
    for i,v in enumerate(pct):ax.text(v+.7,i,f'{v:.1f}',va='center',fontsize=8)
fig.subplots_adjust(left=.11,right=.99,bottom=.19,top=.88,wspace=.65)
fig.text(.11,.035,'Counts follow the supplied dataset overview; they are not assumed to be evaluation-set denominators.',fontsize=8,color='#647286')
save(fig,'figS02_datasets','dataset_overview.xlsx / Sheet1 / C2:C15 and D6:E23')

# 6. Timing: beam sizes and seconds confirmed by the user.
tm=read_xlsx(ROOT/'speed-vs-casanovo/speed_vs_casanovo.xlsx')['Sheet1']
beam=np.arange(1,11); casa=np.array([tm[f'{c}2'] for c in 'BCDEFGHIJK']);roc=np.array([tm[f'{c}3'] for c in 'BCDEFGHIJK'])
fig,axs=plt.subplots(1,2,figsize=(10.5,4.1),gridspec_kw={'width_ratios':[1.3,1]})
for y,color,label in [(casa,BLUE,'Casanovo'),(roc,TEAL,'RocNovo')]:axs[0].plot(beam,y,'o-',color=color,label=label,lw=2,ms=4)
axs[0].set_ylabel('Runtime (s)');axs[0].set_ylim(0,1000);axs[0].legend(frameon=False);panel(axs[0],'A','Inference runtime')
ratio=casa/roc
axs[1].plot(beam,ratio,'o-',color=ORANGE,lw=2,ms=4);axs[1].axhline(1,color=GREY,ls='--',lw=1)
axs[1].set_ylim(0,4.3);axs[1].set_ylabel('Runtime ratio (Casanovo / RocNovo)');panel(axs[1],'B','Relative runtime')
for i in [0,9]:axs[1].annotate(f'{ratio[i]:.2f}×',(beam[i],ratio[i]),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
for ax in axs:ax.set_xticks(beam);ax.set_xlabel('Beam size');grid(ax,'y')
fig.subplots_adjust(left=.08,right=.98,bottom=.2,top=.87,wspace=.35)
fig.text(.08,.025,'Recorded timings; no repeat-run uncertainty or hardware configuration is supplied in the source workbook.',fontsize=8,color='#647286')
save(fig,'fig05_runtime','speed_vs_casanovo.xlsx / Sheet1 / B1:K3; beam size 1–10 and seconds confirmed by user')

(OUT/'sources.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
# Contact sheet for quick visual verification.
thumbs=[]
for name in manifest:
    im=Image.open(OUT/f'{name}.png').convert('RGB')
    im.thumbnail((1000,550))
    tile=Image.new('RGB',(1040,600),'#EEF1F5');tile.paste(im,((1040-im.width)//2,35))
    ImageDraw.Draw(tile).text((20,12),name,fill=INK)
    thumbs.append(tile)
sheet=Image.new('RGB',(2080,1800),'white')
for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*1040,(i//2)*600))
sheet.save(OUT/'overview.png')
print('Saved six figures in PDF, SVG, PNG and overview.png')
