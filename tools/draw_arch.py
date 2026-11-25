# Draws the architecture diagram: python tools/draw_arch.py -> arch.html, then screenshot it at 1920x1080.
X=lambda i:75+260*(i-1); W=210; H=130
BY={1:165,2:385,3:605}
cols={"io":"#334155","conv":"#2563eb","res":"#0f766e","re":"#7c3aed"}
o=[]
def card(c,b,title,sub,tag,kind,par=None,hl=False):
    x=X(c);y=BY[b];col=cols[kind]
    st=f'stroke="#d97706" stroke-width="3"' if hl else 'stroke="#cbd5e1" stroke-width="1.5"'
    o.append(f'<g filter="url(#sh)"><rect x="{x}" y="{y}" width="{W}" height="{H}" rx="12" fill="#fff" {st}/></g>')
    o.append(f'<path d="M{x} {y+12} a12 12 0 0 1 12 -12 h{W-24} a12 12 0 0 1 12 12 v6 h-{W} z" fill="{col}"/>')
    o.append(f'<text x="{x+14}" y="{y+36}" class="t">{title}</text>')
    if par: o.append(f'<text x="{x+W-12}" y="{y+36}" class="p" text-anchor="end">{par}</text>')
    o.append(f'<text x="{x+14}" y="{y+62}" class="s">{sub}</text>')
    o.append(f'<rect x="{x+12}" y="{y+84}" width="{W-24}" height="30" rx="15" fill="{col}1a"/>')
    o.append(f'<text x="{x+W/2}" y="{y+104}" class="g" text-anchor="middle" fill="{col}">{tag}</text>')
def arrow(pts,color="#475569",dash=False,m="a"):
    d="M"+" L".join(f"{a} {b}" for a,b in pts)
    o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.4" {"stroke-dasharray=\"8 6\"" if dash else ""} marker-end="url(#{m})"/>')
# bands
for b,(lab,fill,lc) in {1:("Full resolution  H × W","#f1f5f9","#475569"),2:("1/2 resolution  H/2 × W/2   ·   width w = 32","#eaf2fd","#1d4ed8"),3:("1/4 resolution  H/4 × W/4   ·   width 2w = 64","#fdf1dc","#b45309")}.items():
    y=BY[b]-50
    o.append(f'<rect x="40" y="{y}" width="1840" height="200" rx="16" fill="{fill}"/>')
    o.append(f'<text x="62" y="{y+30}" class="b" fill="{lc}">{lab}</text>')
card(1,1,"Input image x","rainy / hazy","3 × H × W","io")
card(2,1,"PixelUnshuffle ×2","space → channels","12 × H/2 × W/2","re")
card(2,2,"Head","Conv3×3 + ReLU 12→32","32 × H/2 × W/2","conv","3.5K")
card(3,2,"Encoder","2 × ResBlock(32)","e: 32 × H/2 × W/2","res","37.0K")
card(3,3,"Down","Conv3×3, stride 2 32→64","64 × H/4 × W/4","conv","18.5K")
card(4,3,"Bottleneck","6 × ResBlock(64)","64 × H/4 × W/4","res","443.1K",True)
card(5,3,"Up","Conv1×1 + PixelShuffle","32 × H/2 × W/2","re","8.3K")
card(5,2,"Fuse","Concat(up, e) → Conv1×1","32 × H/2 × W/2","conv","2.1K")
card(6,2,"Decoder","2 × ResBlock(32)","32 × H/2 × W/2","res","37.0K")
card(7,2,"Tail","Conv3×3 32→12","12 × H/2 × W/2","conv","3.5K")
card(7,1,"PixelShuffle ×2","channels → space","rain/haze layer R: 3×H×W","re")
card(5,1,"Output B","clean image","3 × H × W","io")
cx=X(6)+W/2
o.append(f'<circle cx="{cx}" cy="230" r="24" fill="#fff" stroke="#7c3aed" stroke-width="2.6"/><text x="{cx}" y="239" text-anchor="middle" font-size="28" font-weight="700" fill="#7c3aed">−</text>')
arrow([(285,230),(335,230)]); arrow([(440,295),(440,385)])
arrow([(545,450),(595,450)]); arrow([(700,515),(700,605)])
arrow([(805,670),(855,670)]); arrow([(1065,670),(1115,670)]); arrow([(1220,605),(1220,515)])
arrow([(805,450),(1115,450)],"#0f766e",True,"g2")
o.append('<text x="960" y="438" class="s" text-anchor="middle" fill="#0f766e">skip connection (e)</text>')
arrow([(1325,450),(1375,450)]); arrow([(1585,450),(1635,450)]); arrow([(1740,385),(1740,295)])
arrow([(1635,230),(1506,230)],"#7c3aed",False,"p"); arrow([(1454,230),(1325,230)],"#7c3aed",False,"p")
arrow([(250,165),(250,137),(cx,137),(cx,204)],"#7c3aed",True,"p")
o.append(f'<text x="1000" y="127" class="s" text-anchor="middle" fill="#7c3aed">Residual path: x goes to the output; the network only predicts R</text>')
# KPI
k=[("0.55 M","parameters (80% in the Bottleneck)"),("8.2 ms","720p · FP16 · about 120 FPS"),("27.64 dB","Outdoor-Rain PSNR  (Rain100L 31.10 / H 26.94)"),("0.858","downstream detection mAP50  (rainy input 0.395)")]
for i,(a,b) in enumerate(k):
    x=40+i*465
    o.append(f'<g filter="url(#sh)"><rect x="{x}" y="810" width="445" height="120" rx="14" fill="#fff" stroke="#cbd5e1" stroke-width="1.5"/></g><rect x="{x}" y="810" width="8" height="120" rx="4" fill="#2563eb"/>')
    o.append(f'<text x="{x+30}" y="868" font-size="44" font-weight="800" fill="#0f172a">{a}</text><text x="{x+30}" y="908" class="s">{b}</text>')
leg=[("io","input / output"),("conv","convolution"),("res","residual blocks"),("re","reshuffle (Pixel[Un]Shuffle / upsample)")]
x=40
for kd,t in leg:
    o.append(f'<rect x="{x}" y="978" width="22" height="22" rx="5" fill="{cols[kd]}"/><text x="{x+32}" y="995" class="s">{t}</text>'); x+=60+len(t)*11
o.append(f'<text x="1880" y="995" class="s" text-anchor="end">ResBlock = Conv3×3 → ReLU → Conv3×3 + identity shortcut   ·   trained on Rain13K + Outdoor-Rain for 300k iters   ·   no GAN</text>')
o.append('<text x="1880" y="1040" class="s" text-anchor="end" fill="#94a3b8">Number at the top right of each card = parameters in that layer</text>')
defs=''.join(f'<marker id="{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{c}"/></marker>' for i,c in [("a","#475569"),("g2","#0f766e"),("p","#7c3aed")])
html=f'''<html><body style="margin:0"><svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080" font-family="Noto Sans, DejaVu Sans, sans-serif">
<defs>{defs}<filter id="sh" x="-5%" y="-5%" width="110%" height="120%"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.12"/></filter></defs>
<style>.t{{font-size:19px;font-weight:700;fill:#0f172a}}.s{{font-size:14.5px;fill:#475569}}.p{{font-size:13px;font-weight:600;fill:#64748b}}.g{{font-size:14px;font-weight:600}}.b{{font-size:17px;font-weight:700}}</style>
<rect width="1920" height="1080" fill="#fff"/>
<text x="40" y="52" font-size="36" font-weight="800" fill="#0f172a">LightDerainNet: one lightweight model for both deraining and dehazing</text>
<text x="40" y="84" font-size="18" fill="#64748b">Lightweight U-shaped residual CNN · PixelUnshuffle to 1/2 resolution, bottleneck at 1/4 · predicts the rain/haze layer and subtracts it (B = x − R)</text>
{"".join(o)}</svg></body></html>'''
open("arch.html","w").write(html)
