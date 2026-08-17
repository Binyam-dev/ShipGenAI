from __future__ import annotations

import json, math, random, subprocess, textwrap
from pathlib import Path
import numpy as np
import requests, soundfile as sf
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from kokoro import KPipeline

ROOT=Path(__file__).resolve().parent
BUILD=ROOT/'build'; DIST=ROOT/'dist'; BUILD.mkdir(parents=True,exist_ok=True); DIST.mkdir(parents=True,exist_ok=True)
W,H,FPS=1080,1920,30
VOICE='am_onyx'; SPEED=.90; PAUSE=.22
NASA_IMG='https://svs.gsfc.nasa.gov/vis/a010000/a014300/a014380/Icy_Rogue_Planet_Final.jpg'
NASA_VID='https://svs.gsfc.nasa.gov/vis/a020000/a020300/a020315/ROMAN_MicroL_RogueP_4k_30fps_h264.mp4'
SEG=[
'In the Milky Way, not every planet has a sun.',
'Rogue planets travel through interstellar space, gravitationally bound to no star.',
'Some were likely ejected from young planetary systems by violent gravitational encounters. Others may have formed alone.',
'They are almost impossible to see directly because they are cold and dark.',
'But gravity gives them away. If a rogue planet passes in front of a distant star, its gravity bends and magnifies the star’s light.',
'Astronomers call the effect microlensing. For small worlds, the brightening can last only hours or days, then disappear.',
'NASA says recent research suggests rogue planets could outnumber star-bound worlds by about six to one.',
'That would mean trillions of wandering worlds in our galaxy.',
'Their surfaces would be brutally cold, but internal heat and thick ice could preserve warmer layers below.',
'Some scientists have even considered whether hidden oceans could survive beneath frozen crusts.',
'NASA’s Roman Space Telescope is designed to find hundreds of these worlds through microlensing.',
'A universe full of planets with no sunrise, no sunset, and no home star. Invisible to our eyes, but not to gravity.'
]

def run(cmd):
    print('+',' '.join(map(str,cmd)),flush=True); subprocess.run(list(map(str,cmd)),check=True)

def dur(p):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nk=1:nw=1',str(p)],text=True).strip())

def font(size,bold=False):
    p='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    return ImageFont.truetype(p,size)

def download(url,p):
    if p.exists() and p.stat().st_size>20000:return
    r=requests.get(url,timeout=120,headers={'User-Agent':'BehindBillions/1.0'}); r.raise_for_status(); p.write_bytes(r.content)
    if p.stat().st_size<20000: raise RuntimeError('download too small '+url)

def stars(seed,nebula=True):
    rng=np.random.default_rng(seed); y=np.linspace(0,1,H)[:,None]; a=np.zeros((H,W,3),dtype=np.float32)
    a[:,:,0]=2+6*(1-y); a[:,:,1]=5+10*(1-y); a[:,:,2]=14+20*(1-y)
    if nebula:
        yy,xx=np.mgrid[0:H,0:W]; ang=-.56; cx,cy=W*.48,H*.48
        d=(xx-cx)*math.sin(ang)+(yy-cy)*math.cos(ang); band=np.exp(-(d/(H*.105))**2)
        noise=rng.normal(0,1,(H,W)); ni=Image.fromarray(np.uint8(np.clip((noise+3)*36,0,255))).filter(ImageFilter.GaussianBlur(42)); n=np.asarray(ni)/255.
        glow=band*(.32+.9*n); a[:,:,0]+=58*glow; a[:,:,1]+=34*glow; a[:,:,2]+=82*glow
    im=Image.fromarray(np.clip(a,0,255).astype(np.uint8),'RGB'); d=ImageDraw.Draw(im); rr=random.Random(seed)
    for _ in range(1100):
        x=rr.randrange(W); yy=rr.randrange(H); b=rr.randrange(90,256); r=1 if rr.random()<.94 else 2
        d.ellipse((x-r,yy-r,x+r,yy+r),fill=(b,b,min(255,b+rr.randrange(0,34))))
    return im

def nasa_vertical(src,darken=.75):
    im=Image.open(src).convert('RGB'); bg=im.copy(); s=max(W/bg.width,H/bg.height); bg=bg.resize((int(bg.width*s),int(bg.height*s)),Image.Resampling.LANCZOS)
    bg=bg.crop(((bg.width-W)//2,(bg.height-H)//2,(bg.width-W)//2+W,(bg.height-H)//2+H)).filter(ImageFilter.GaussianBlur(24)); bg=ImageEnhance.Brightness(bg).enhance(.40)
    fg=im.copy(); s=min((W*1.03)/fg.width,(H*.69)/fg.height); fg=fg.resize((int(fg.width*s),int(fg.height*s)),Image.Resampling.LANCZOS); fg=ImageEnhance.Brightness(fg).enhance(darken)
    c=bg.convert('RGBA'); c.alpha_composite(fg.convert('RGBA'),((W-fg.width)//2,int(H*.23))); return c.convert('RGB')

def ejection(im):
    l=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(l); sx,sy=int(W*.24),int(H*.45)
    for r,a in [(62,45),(40,100),(21,235)]:d.ellipse((sx-r,sy-r,sx+r,sy+r),fill=(255,190,90,a))
    for r in [130,210,300]:d.arc((sx-r,sy-r,sx+r,sy+r),250,100,fill=(180,190,210,70),width=3)
    ex,ey=int(W*.78),int(H*.58); d.line((sx+185,sy+65,ex,ey),fill=(225,176,68,135),width=4); d.ellipse((ex-82,ey-82,ex+82,ey+82),fill=(8,13,23,255),outline=(75,145,205,190),width=5)
    return Image.alpha_composite(im.convert('RGBA'),l).convert('RGB')

def lens(im):
    cx,cy=W//2,int(H*.43); l=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(l)
    for r,a,w in [(270,60,18),(250,125,10),(235,235,4)]:d.ellipse((cx-r,cy-r,cx+r,cy+r),outline=(255,210,135,a),width=w)
    d.ellipse((cx-20,cy-20,cx+20,cy+20),fill=(255,248,220,255)); pts=[]
    for x in range(100,W-100,5):
        u=(x-W/2)/170; pts.append((x,int(H*.77-130/(1+u*u))))
    d.line(pts,fill=(225,176,68,235),width=4); return Image.alpha_composite(im.convert('RGBA'),l).convert('RGB')

def ice(im,seed):
    rr=random.Random(seed); l=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(l); d.rectangle((0,int(H*.55),W,H),fill=(4,18,36,230))
    for y in range(int(H*.62),H,70):d.rectangle((0,y,W,y+30),fill=(5,38,65,55))
    for _ in range(90):
        x=rr.randrange(W); y=rr.randrange(int(H*.53),int(H*.72)); d.line((x,y,x+rr.randrange(-95,95),y+rr.randrange(25,130)),fill=(105,185,225,110),width=2)
    for x in [250,560,830]:
        for k in range(4):d.ellipse((x-60-k*20,int(H*.85)-70-k*35,x+60+k*20,int(H*.85)+70+k*35),fill=(20,135,185,18))
    return Image.alpha_composite(im.convert('RGBA'),l).convert('RGB')

def vignette(im):
    a=np.asarray(im).astype(np.float32); yy,xx=np.mgrid[0:H,0:W]; dx=(xx-W/2)/(W/2); dy=(yy-H/2)/(H/2); v=np.clip(1-.34*(dx*dx+dy*dy),.58,1)[...,None]
    return Image.fromarray(np.clip(a*v,0,255).astype(np.uint8))

def caption(im,i,text,nasa=False):
    im=im.convert('RGBA'); o=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(o)
    d.text((54,60),'BEHIND BILLIONS',font=font(28,True),fill=(225,176,68,255)); d.text((54,99),'THE UNSEEN IS REAL',font=font(15),fill=(205,210,220,235))
    d.text((W-330,64),'NASA + ORIGINAL VISUALS',font=font(14),fill=(185,190,200,210))
    lines=textwrap.wrap(text,width=34); fs=42 if len(lines)<=3 else 37; f=font(fs,True); bb=[d.textbbox((0,0),x,font=f) for x in lines]; th=sum(b[3]-b[1] for b in bb)+11*(len(lines)-1); top=H-205-th
    d.rounded_rectangle((52,top-30,W-52,H-145),radius=28,fill=(0,0,0,180),outline=(225,176,68,95),width=2); y=top
    for line,b in zip(lines,bb):
        tw=b[2]-b[0]; hh=b[3]-b[1]; d.text(((W-tw)/2,y),line,font=f,fill=(250,250,250,255),stroke_width=2,stroke_fill=(0,0,0,230)); y+=hh+11
    if i==0:
        d.rounded_rectangle((55,1115,1025,1545),radius=34,fill=(0,0,0,160)); d.text((92,1160),'THERE ARE',font=font(72,True),fill='white'); d.text((92,1252),'WORLDS',font=font(104,True),fill='white'); d.text((92,1380),'WITH NO SUN',font=font(71,True),fill=(225,176,68,255))
    return Image.alpha_composite(im,o).convert('RGB')

def scene(i,nasa):
    if i in (0,1,3,6,10,11):im=nasa_vertical(nasa,.63 if i in (3,11) else .78)
    else:
        im=stars(5000+i,i not in (8,9));
        if i==2:im=ejection(im)
        elif i in (4,5):im=lens(im)
        elif i==7:
            rr=random.Random(7007); l=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(l)
            for _ in range(1600):
                x=int(rr.gauss(W*.5,W*.24)); y=int(rr.gauss(H*.48,H*.18))
                if 0<=x<W and 0<=y<H:
                    r=1 if rr.random()<.88 else 2; a=rr.randrange(70,210); d.ellipse((x-r,y-r,x+r,y+r),fill=(235,195,105,a))
            im=Image.alpha_composite(im.convert('RGBA'),l).convert('RGB')
        elif i in (8,9):im=ice(im,8000+i)
    return vignette(im)

def overlay_png(i,text,p):
    im=Image.new('RGBA',(W,H),(0,0,0,0)); im=caption(im,i,text,True); im.save(p)

def main():
    ni=BUILD/'nasa_rogue.jpg'; nv=BUILD/'nasa_microlensing.mp4'; download(NASA_IMG,ni)
    try:download(NASA_VID,nv)
    except Exception as e:print('NASA video fallback:',e); nv.unlink(missing_ok=True)

    pipe=KPipeline(lang_code='a'); voices=[]; ds=[]
    for i,t in enumerate(SEG):
        parts=[np.asarray(a,dtype=np.float32) for _,_,a in pipe(t,voice=VOICE,speed=SPEED,split_pattern=r'\n+')]
        if not parts:raise RuntimeError('Kokoro produced no audio')
        a=np.concatenate(parts); peak=float(np.max(np.abs(a))) or 1; a=np.clip(a*min(1,.92/peak),-1,1); raw=BUILD/f'raw_{i:02d}.wav'; out=BUILD/f'voice_{i:02d}.wav'; sf.write(raw,a,24000)
        run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',raw,'-af','highpass=f=65,lowpass=f=10500,equalizer=f=120:t=q:w=1:g=2.4,equalizer=f=240:t=q:w=1.1:g=1.2,equalizer=f=3500:t=q:w=1.2:g=-0.8,acompressor=threshold=0.13:ratio=2.6:attack=8:release=120:makeup=1.5,alimiter=limit=0.94','-ar','48000','-ac','1',out]); voices.append(out); ds.append(dur(out)+PAUSE)

    inp=[]; flt=[]
    for i,p in enumerate(voices):inp+=['-i',p]; flt.append(f'[{i}:a]apad=pad_dur={PAUSE},atrim=duration={ds[i]:.3f}[a{i}]')
    labels=''.join(f'[a{i}]' for i in range(len(voices))); narration=BUILD/'narration.wav'; run(['ffmpeg','-y','-hide_banner','-loglevel','error',*inp,'-filter_complex',';'.join(flt)+f';{labels}concat=n={len(voices)}:v=0:a=1[n]','-map','[n]','-ar','48000',narration]); total=sum(ds)

    clips=[]
    for i,(t,d) in enumerate(zip(SEG,ds)):
        out=BUILD/f'clip_{i:02d}.mp4'
        if i in (4,5) and nv.exists():
            ov=BUILD/f'ov_{i:02d}.png'; overlay_png(i,t,ov); start=0 if i==4 else 4.5
            fc='[0:v]split=2[bg][fg];[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,gblur=sigma=35,eq=brightness=-0.22[bg2];[fg]scale=1080:-2:force_original_aspect_ratio=decrease[fg2];[bg2][fg2]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2[mid];[mid][1:v]overlay=0:0,format=yuv420p[out]'
            run(['ffmpeg','-y','-hide_banner','-loglevel','error','-stream_loop','-1','-ss',f'{start:.2f}','-i',nv,'-loop','1','-i',ov,'-t',f'{d:.3f}','-filter_complex',fc,'-map','[out]','-an','-r',str(FPS),'-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p',out])
        else:
            im=caption(scene(i,ni),i,t); img=BUILD/f'scene_{i:02d}.jpg'; im.save(img,quality=93,optimize=True); frames=max(1,math.ceil(d*FPS)); vf=f"zoompan=z='min(zoom+0.00042,1.065)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}:fps={FPS},format=yuv420p"
            run(['ffmpeg','-y','-hide_banner','-loglevel','error','-loop','1','-i',img,'-t',f'{d:.3f}','-vf',vf,'-an','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p',out])
        clips.append(out)
    cf=BUILD/'concat.txt'; cf.write_text(''.join(f"file '{p.as_posix()}'\n" for p in clips)); visual=BUILD/'visual.mp4'; run(['ffmpeg','-y','-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',cf,'-c','copy',visual])

    amb=BUILD/'ambient.wav'; af=f"sine=frequency=49:sample_rate=48000:duration={total:.3f},volume=0.018[a0];sine=frequency=73.5:sample_rate=48000:duration={total:.3f},volume=0.010[a1];sine=frequency=147:sample_rate=48000:duration={total:.3f},volume=0.004[a2];anoisesrc=color=pink:sample_rate=48000:duration={total:.3f},highpass=f=90,lowpass=f=1000,volume=0.0035[n];[a0][a1][a2][n]amix=inputs=4:normalize=0,afade=t=in:st=0:d=2.2,afade=t=out:st={max(0,total-3):.3f}:d=3[amb]"; run(['ffmpeg','-y','-hide_banner','-loglevel','error','-filter_complex',af,'-map','[amb]',amb])
    mix=BUILD/'mix.wav'; run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',narration,'-i',amb,'-filter_complex','[0:a]loudnorm=I=-15.5:TP=-1.5:LRA=8[v];[1:a]volume=.70[a];[v][a]amix=inputs=2:duration=first:normalize=0,alimiter=limit=.95[o]','-map','[o]','-ar','48000','-ac','2',mix])
    master=DIST/'BehindBillions_001_ThereAreWorldsWithNoSun.mp4'; run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',visual,'-i',mix,'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','160k','-ar','48000','-movflags','+faststart','-shortest',master])
    web=DIST/'BehindBillions_001_ThereAreWorldsWithNoSun_web.mp4'; run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',master,'-c:v','libx264','-preset','veryfast','-b:v','2200k','-maxrate','2600k','-bufsize','5200k','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-movflags','+faststart',web])
    q=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration,size,bit_rate:stream=codec_name,codec_type,width,height,r_frame_rate,sample_rate,channels','-of','json',web],text=True)); (DIST/'qc.json').write_text(json.dumps(q,indent=2)); (DIST/'credits.txt').write_text('NASA Goddard: Icy Earth-mass Rogue Planet and Roman microlensing animation. Narration: original Kokoro am_onyx voice; no real-person imitation. Additional graphics and ambient score: original procedural production.\n')
    v=[s for s in q['streams'] if s.get('codec_type')=='video']; a=[s for s in q['streams'] if s.get('codec_type')=='audio']; dd=float(q['format']['duration']); assert v and v[0]['width']==1080 and v[0]['height']==1920 and v[0]['codec_name']=='h264'; assert a and a[0]['codec_name']=='aac' and a[0]['sample_rate']=='48000'; assert 45<=dd<=120 and web.stat().st_size>5_000_000; print('QC PASS',json.dumps(q,indent=2))

if __name__=='__main__':main()
