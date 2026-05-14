"""
main_webcam_drowsiness.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real-Time Driver Drowsiness Detection — FINAL SYSTEM
Aesthetic: Dark automotive HUD — inspired by ADAS dashboards.
           Muted carbon background, electric-cyan accents,
           amber warnings, deep-red critical alerts.
           Segmented arc gauges, glowing indicators,
           scanline texture, monospaced telemetry readouts.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import csv, pathlib, sys, time, threading, collections, math
from typing import Optional, Deque

import cv2, mediapipe as mp, numpy as np

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
from onnx_inference import DrowsinessInferenceEngine


# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
class CFG:
    EYE_ONNX   = pathlib.Path("modals/onnx/eye_model3.onnx")
    MOUTH_ONNX = pathlib.Path("modals/onnx/mouth_model.onnx")
    LOG_FILE   = pathlib.Path("logs/session3.csv")
    CAM_INDEX  = 0
    CAM_W, CAM_H = 640, 480

    EAR_THRESH = 0.21
    MAR_THRESH = 0.65
    ML_EYE_THRESH   = 0.40
    ML_MOUTH_THRESH = 0.40
    EAR_W, ML_W     = 0.70, 0.30
    FUSED_THRESH     = 0.45

    BLINK_MIN_SEC  = 0.35
    ALERT_MIN_SEC  = 0.35
    DROWSY_MIN_SEC = 2.00

    YAWN_MIN_SEC       = 0.50
    YAWN_DROWSY_COUNT  = 1
    YAWN_DROWSY_WINDOW = 100.0

    PERCLOS_WINDOW_SEC    = 30.0
    PERCLOS_DROWSY_THRESH = 0.35
    ALERT_LINGER = 2.5

    BEEP_HZ = 1000; BEEP_MS = 400; BEEP_INTERVAL = 3.0
    MIN_DETECT = 0.5; MIN_TRACK = 0.5
    WIN_TITLE  = "DRIVEWATCH — Drowsiness Monitor"
    FLASH_EVERY = 10


# ═══════════════════════════════════════════════════════════════════
#  LANDMARK INDICES
# ═══════════════════════════════════════════════════════════════════
_L_EAR  = [362,385,387,263,373,380]
_R_EAR  = [33,160,158,133,153,144]
_L_RING = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
_R_RING = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
_M_MAR  = [78,13,308,14]
_M_CROP = [61,146,91,181,84,17,314,405,321,375,291,185,40,39,37,0,267,269,270,409]


# ═══════════════════════════════════════════════════════════════════
#  PALETTE  (BGR)
# ═══════════════════════════════════════════════════════════════════
C = {
    # backgrounds
    "void"    : (8,    8,   10),
    "panel"   : (16,  18,   22),
    "border"  : (35,  42,   52),
    "groove"  : (25,  30,   38),
    # status
    "normal"  : (180, 230,  80),    # lime-green
    "alert"   : (0,   185, 255),    # amber-orange
    "drowsy"  : (30,   30, 230),    # vivid red
    "yawn"    : (200, 210,   0),    # cyan-gold
    # accents
    "cyan"    : (230, 210,   0),    # electric cyan (primary accent)
    "cyan_dim": (100,  90,   0),
    "white"   : (255, 255, 255),
    "silver"  : (180, 185, 190),
    "dim"     : (80,   85,  95),
    "dark"    : (30,   35,  42),
    # gauge fills
    "g_ear"   : (60,  220, 120),
    "g_mar"   : (60,  200, 220),
    "g_ml"    : (200, 140, 255),
    "g_fused" : (60,  210, 255),
    "g_pclos" : (0,   160, 255),
    # roi
    "eye_ok"  : (60,  220, 120),
    "eye_bad" : (30,   30, 230),
    "mth_ok"  : (60,  210, 200),
    "mth_bad" : (0,   185, 255),
    "mesh"    : (30,   36,  46),
}


# ═══════════════════════════════════════════════════════════════════
#  STATUS
# ═══════════════════════════════════════════════════════════════════
class Status:
    NORMAL="NORMAL"; ALERT="ALERT"; DROWSY="DROWSY"; YAWN="YAWN"
    _col = {NORMAL:C["normal"],ALERT:C["alert"],
            DROWSY:C["drowsy"],YAWN:C["yawn"]}
    @classmethod
    def colour(cls,s): return cls._col.get(s,C["white"])


# ═══════════════════════════════════════════════════════════════════
#  BLINK FILTER
# ═══════════════════════════════════════════════════════════════════
class BlinkFilter:
    def __init__(self):
        self._since: Optional[float]=None
        self.duration=0.0; self.closed_valid=False
    def update(self,raw:bool):
        now=time.time()
        if raw:
            if self._since is None: self._since=now
            self.duration=now-self._since
            self.closed_valid=self.duration>=CFG.BLINK_MIN_SEC
        else:
            self._since=None; self.duration=0.0; self.closed_valid=False
    def reset(self):
        self._since=None; self.duration=0.0; self.closed_valid=False


# ═══════════════════════════════════════════════════════════════════
#  YAWN COUNTER
# ═══════════════════════════════════════════════════════════════════
class YawnCounter:
    def __init__(self):
        self._times:Deque=collections.deque()
        self._in=False; self._start:Optional[float]=None
        self.cur_dur=0.0; self.just_confirmed=False
    def update(self,open_:bool):
        now=time.time(); self.just_confirmed=False
        if open_:
            if not self._in: self._in=True; self._start=now
            self.cur_dur=now-self._start
        else:
            if self._in and self.cur_dur>=CFG.YAWN_MIN_SEC:
                self._times.append(now); self.just_confirmed=True
            self._in=False; self._start=None; self.cur_dur=0.0
        cutoff=now-CFG.YAWN_DROWSY_WINDOW
        while self._times and self._times[0]<cutoff: self._times.popleft()
    @property
    def count(self): return len(self._times)
    @property
    def drowsy(self): return self.count>=CFG.YAWN_DROWSY_COUNT
    def reset(self):
        self._times.clear(); self._in=False; self._start=None; self.cur_dur=0.0


# ═══════════════════════════════════════════════════════════════════
#  PERCLOS
# ═══════════════════════════════════════════════════════════════════
class Perclos:
    def __init__(self):
        self._s:Deque=collections.deque()
    def update(self,v:bool):
        now=time.time(); self._s.append((now,v))
        c=now-CFG.PERCLOS_WINDOW_SEC
        while self._s and self._s[0][0]<c: self._s.popleft()
    @property
    def value(self):
        if not self._s: return 0.0
        return sum(1 for _,v in self._s if v)/len(self._s)
    @property
    def n(self): return len(self._s)
    def reset(self): self._s.clear()


# ═══════════════════════════════════════════════════════════════════
#  GEOMETRY
# ═══════════════════════════════════════════════════════════════════
def _px(lm,i,w,h): p=lm[i]; return np.array([p.x*w,p.y*h])

def ear_(lm,idx,w,h):
    p=[_px(lm,i,w,h) for i in idx]
    return (np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2*np.linalg.norm(p[0]-p[3])+1e-6)

def mar_(lm,idx,w,h):
    p=[_px(lm,i,w,h) for i in idx]
    return np.linalg.norm(p[1]-p[3])/(np.linalg.norm(p[0]-p[2])+1e-6)

def crop_(frame,lm,idx,w,h,pad):
    xs=[lm[i].x*w for i in idx]; ys=[lm[i].y*h for i in idx]
    bw=max(xs)-min(xs); bh_=max(ys)-min(ys)
    x0=max(0,int(min(xs)-bw*pad)); y0=max(0,int(min(ys)-bh_*pad))
    x1=min(w,int(max(xs)+bw*pad)); y1=min(h,int(max(ys)+bh_*pad))
    c=frame[y0:y1,x0:x1]; return c if c.size>0 else None


# ═══════════════════════════════════════════════════════════════════
#  FUSION
# ═══════════════════════════════════════════════════════════════════
def fuse_eye(ear_v,ml_v):
    ear_s=1.0 if ear_v<CFG.EAR_THRESH else 0.0
    if ml_v<CFG.ML_EYE_THRESH: return ear_s,ear_s>=0.5
    f=CFG.EAR_W*ear_s+CFG.ML_W*ml_v
    return f,f>=CFG.FUSED_THRESH

def fuse_mouth(mar_v,ml_v):
    return mar_v>CFG.MAR_THRESH and ml_v>=CFG.ML_MOUTH_THRESH


# ═══════════════════════════════════════════════════════════════════
#  SOUND
# ═══════════════════════════════════════════════════════════════════
class Sound:
    def __init__(self): self._t=0.0; self._lk=threading.Lock()
    def beep(self):
        n=time.time()
        with self._lk:
            if n-self._t<CFG.BEEP_INTERVAL: return
            self._t=n
        threading.Thread(target=self._p,daemon=True).start()
    @staticmethod
    def _p():
        try:
            import winsound; winsound.Beep(CFG.BEEP_HZ,CFG.BEEP_MS)
        except: pass


# ═══════════════════════════════════════════════════════════════════
#  DRAWING PRIMITIVES
# ═══════════════════════════════════════════════════════════════════
def _blend(frame,x0,y0,x1,y1,col,alpha=0.82):
    x0=max(0,x0);y0=max(0,y0)
    x1=min(frame.shape[1],x1);y1=min(frame.shape[0],y1)
    if x1<=x0 or y1<=y0: return
    roi=frame[y0:y1,x0:x1]
    bg=np.full_like(roi,col,dtype=np.uint8)
    cv2.addWeighted(bg,alpha,roi,1-alpha,0,roi)
    frame[y0:y1,x0:x1]=roi

def _t(frame,txt,x,y,sc=0.42,col=None,th=1,bold=False):
    cv2.putText(frame,txt,(x,y),
        cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX,
        sc,col or C["silver"],th,cv2.LINE_AA)

def _line(frame,p1,p2,col,thick=1):
    cv2.line(frame,p1,p2,col,thick,cv2.LINE_AA)

def _rect(frame,x0,y0,x1,y1,col,thick=1):
    cv2.rectangle(frame,(x0,y0),(x1,y1),col,thick,cv2.LINE_AA)

def _circle(frame,cx,cy,r,col,fill=-1):
    cv2.circle(frame,(cx,cy),r,col,fill,cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════
#  SCANLINE OVERLAY  (gives the screen a CRT / HUD texture)
# ═══════════════════════════════════════════════════════════════════
_SCANLINES = None
def _build_scanlines(h,w):
    global _SCANLINES
    sl=np.zeros((h,w,3),dtype=np.uint8)
    for y in range(0,h,3):
        sl[y,:]=6
    _SCANLINES=sl

def apply_scanlines(frame):
    if _SCANLINES is None or _SCANLINES.shape!=frame.shape: return
    cv2.addWeighted(frame,1.0,_SCANLINES,0.35,0,frame)


# ═══════════════════════════════════════════════════════════════════
#  ARC GAUGE
# ═══════════════════════════════════════════════════════════════════
def draw_arc_gauge(frame, cx, cy, r, value, max_val,
                   thresh, col_fill, label, val_str,
                   start_deg=200, sweep=140):
    """
    Segmented arc gauge — automotive instrument cluster style.
    Draws tick segments from start_deg sweeping clockwise.
    """
    n_segs = 20
    seg_sweep = sweep / n_segs
    filled = int(n_segs * min(value/max_val, 1.0))
    thresh_seg = int(n_segs * min(thresh/max_val, 1.0))

    for i in range(n_segs):
        a_start = math.radians(start_deg + i * seg_sweep + 2)
        a_end   = math.radians(start_deg + (i+1) * seg_sweep - 2)
        r_out, r_in = r, r - 9

        pts_out = [
            (int(cx + r_out*math.cos(a_start)),
             int(cy + r_out*math.sin(a_start))),
            (int(cx + r_out*math.cos(a_end)),
             int(cy + r_out*math.sin(a_end))),
        ]
        pts_in = [
            (int(cx + r_in*math.cos(a_end)),
             int(cy + r_in*math.sin(a_end))),
            (int(cx + r_in*math.cos(a_start)),
             int(cy + r_in*math.sin(a_start))),
        ]
        pts = np.array([pts_out[0],pts_out[1],
                        pts_in[0], pts_in[1]], np.int32)

        if i < filled:
            # Colour transitions: green→amber at threshold
            if i >= thresh_seg:
                seg_col = C["alert"]
            else:
                seg_col = col_fill
        else:
            seg_col = C["groove"]

        cv2.fillPoly(frame, [pts], seg_col)

    # Centre text
    _t(frame, label,   cx-18, cy+5,  0.36, C["dim"])
    _t(frame, val_str, cx-22, cy+18, 0.50, C["silver"], bold=True)


# ═══════════════════════════════════════════════════════════════════
#  ROUNDED RECT
# ═══════════════════════════════════════════════════════════════════
def _rounded(frame,x0,y0,x1,y1,r,col,fill=True,thick=1):
    if fill:
        cv2.rectangle(frame,(x0+r,y0),(x1-r,y1),col,-1)
        cv2.rectangle(frame,(x0,y0+r),(x1,y1-r),col,-1)
        for cx,cy in[(x0+r,y0+r),(x1-r,y0+r),(x0+r,y1-r),(x1-r,y1-r)]:
            cv2.circle(frame,(cx,cy),r,col,-1,cv2.LINE_AA)
    else:
        cv2.rectangle(frame,(x0+r,y0),(x1-r,y0),col,thick)
        cv2.rectangle(frame,(x0+r,y1),(x1-r,y1),col,thick)
        cv2.rectangle(frame,(x0,y0+r),(x0,y1-r),col,thick)
        cv2.rectangle(frame,(x1,y0+r),(x1,y1-r),col,thick)
        for cx,cy,sa,ea in[
            (x0+r,y0+r,180,270),(x1-r,y0+r,270,360),
            (x1-r,y1-r,0,90),  (x0+r,y1-r,90,180)]:
            cv2.ellipse(frame,(cx,cy),(r,r),0,sa,ea,col,thick,cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════
#  HORIZONTAL BAR  (slim, with glow effect)
# ═══════════════════════════════════════════════════════════════════
def _hbar(frame,x,y,bw,bh,val,max_v,thresh,col,label,val_str):
    # track
    _rounded(frame,x,y,x+bw,y+bh,3,C["groove"])
    # fill
    fill=int(np.clip(val/max_v,0,1)*bw)
    if fill>6:
        _rounded(frame,x,y,x+fill,y+bh,3,col)
        # subtle glow: lighter stripe on top 2px
        glow=tuple(min(255,v+60) for v in col)
        cv2.rectangle(frame,(x+4,y+1),(x+fill-2,y+2),glow,-1)
    # threshold tick
    tx=x+int(np.clip(thresh/max_v,0,1)*bw)
    _line(frame,(tx,y-3),(tx,y+bh+3),C["alert"],2)
    # labels
    _t(frame,label,x,y-4,0.35,C["dim"])
    _t(frame,val_str,x+bw+4,y+bh-2,0.38,C["silver"])


# ═══════════════════════════════════════════════════════════════════
#  STATUS PILL  (top-centre)
# ═══════════════════════════════════════════════════════════════════
def draw_status_pill(frame,status,flash_on,fw,fh):
    labels={
        Status.NORMAL:"◉  NORMAL",
        Status.ALERT :"⚠  ALERT",
        Status.DROWSY:"⚡ DROWSY",
        Status.YAWN  :"〜 YAWN",
    }
    label=labels.get(status,status)
    col=Status.colour(status)

    if status==Status.DROWSY and not flash_on: return

    pw,ph=260,38
    px=(fw-pw)//2; py=8
    _rounded(frame,px,py,px+pw,py+ph,12,C["panel"])
    _rounded(frame,px,py,px+pw,py+ph,12,col,fill=False,thick=2)

    # glow strip at top
    glow=tuple(min(255,v+40) for v in col)
    cv2.rectangle(frame,(px+14,py+2),(px+pw-14,py+3),glow,-1)

    ts=cv2.getTextSize(label,cv2.FONT_HERSHEY_DUPLEX,0.72,2)[0]
    cv2.putText(frame,label,
                ((fw-ts[0])//2,py+ph//2+ts[1]//2),
                cv2.FONT_HERSHEY_DUPLEX,0.72,col,2,cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════
#  HEADER BAR
# ═══════════════════════════════════════════════════════════════════
def draw_header(frame,fps,fw,fh):
    _blend(frame,0,0,fw,46,C["panel"],0.90)
    _line(frame,(0,46),(fw,46),C["border"],1)

    # Logo / title
    _t(frame,"DRIVEWATCH",10,30,0.70,C["cyan"],2,bold=True)
    _t(frame,"DRIVER SAFETY SYSTEM",140,20,0.32,C["dim"])
    _t(frame,"v2.0",140,32,0.30,C["cyan_dim"])

    # FPS — right aligned
    fps_col=C["normal"] if fps>25 else C["alert"] if fps>15 else C["drowsy"]
    _t(frame,f"{fps:.0f}",fw-62,30,0.65,fps_col,2,bold=True)
    _t(frame,"FPS",fw-30,30,0.40,C["dim"])

    # Time
    ts=time.strftime("%H:%M:%S")
    _t(frame,ts,fw//2-28,32,0.42,C["dim"])


# ═══════════════════════════════════════════════════════════════════
#  LEFT PANEL  (metric gauges)
# ═══════════════════════════════════════════════════════════════════
def draw_left_panel(frame,ear_avg,mar,ml_eye,ml_mouth,fused,perclos,fw,fh):
    pw=160; px=8; py=54; ph=fh-py-8
    _blend(frame,px,py,px+pw,py+ph,C["panel"],0.88)
    _rounded(frame,px,py,px+pw,py+ph,6,C["border"],fill=False)

    # Section header
    _t(frame,"TELEMETRY",px+8,py+16,0.38,C["cyan"],1,bold=True)
    _line(frame,(px+8,py+20),(px+pw-8,py+20),C["border"],1)

    # Arc gauges row
    gy=py+68
    draw_arc_gauge(frame,px+42,gy,32, ear_avg,0.5,CFG.EAR_THRESH,
                   C["g_ear"],"EAR",f"{ear_avg:.2f}")
    draw_arc_gauge(frame,px+118,gy,32, mar,1.2,CFG.MAR_THRESH,
                   C["g_mar"],"MAR",f"{mar:.2f}")

    # Slim bars — ML + fused + PERCLOS
    bx=px+10; bw=pw-20; bh=8
    metrics=[
        (ml_eye,  1.0,CFG.ML_EYE_THRESH,  C["g_ml"],   "ML EYE",  f"{ml_eye:.2f}"),
        (ml_mouth,1.0,CFG.ML_MOUTH_THRESH, C["g_ml"],   "ML MOUTH",f"{ml_mouth:.2f}"),
        (fused,   1.0,CFG.FUSED_THRESH,    C["g_fused"],"FUSED",   f"{fused:.2f}"),
        (perclos, 1.0,CFG.PERCLOS_DROWSY_THRESH,C["g_pclos"],"PERCLOS",f"{perclos*100:.0f}%"),
    ]
    by=gy+46
    for val,mx,th,col,lbl,vstr in metrics:
        _hbar(frame,bx,by,bw,bh,val,mx,th,col,lbl,vstr)
        by+=26

    # Raw EAR values
    by+=8
    _t(frame,"EAR-L",bx,by,0.34,C["dim"])
    _t(frame,"EAR-R",bx+75,by,0.34,C["dim"])
    by+=14
    cl=C["eye_bad"] if ear_avg<CFG.EAR_THRESH else C["eye_ok"]
    _t(frame,f"{ear_avg:.3f}",bx,by,0.46,cl,bold=True)
    _t(frame,f"{mar:.3f}",bx+75,by,0.46,
       C["mth_bad"] if mar>CFG.MAR_THRESH else C["mth_ok"],bold=True)


# ═══════════════════════════════════════════════════════════════════
#  RIGHT PANEL  (alerts + yawn counter + blink timer)
# ═══════════════════════════════════════════════════════════════════
def draw_right_panel(frame,blink_dur,closed_valid,
                     yawn_count,yawn_dur,yawn_drowsy,
                     drowsy_until,yawn_until,
                     perclos_val,fw,fh):
    pw=160; px=fw-pw-8; py=54; ph=fh-py-8
    _blend(frame,px,py,px+pw,py+ph,C["panel"],0.88)
    _rounded(frame,px,py,px+pw,py+ph,6,C["border"],fill=False)

    _t(frame,"ALERTS",px+8,py+16,0.38,C["cyan"],1,bold=True)
    _line(frame,(px+8,py+20),(px+pw-8,py+20),C["border"],1)

    now=time.time()

    # ── Blink / closure indicator ─────────────────────────────────
    _t(frame,"EYE CLOSURE",px+8,py+36,0.33,C["dim"])

    # Ring timer
    cx=px+pw//2; cy=py+80; r=30
    _circle(frame,cx,cy,r,C["groove"])
    # draw arc proportional to blink duration / DROWSY_MIN_SEC
    frac=min(blink_dur/CFG.DROWSY_MIN_SEC,1.0)
    if frac>0:
        arc_col=(C["drowsy"] if blink_dur>=CFG.DROWSY_MIN_SEC
                 else C["alert"] if blink_dur>=CFG.ALERT_MIN_SEC
                 else C["dim"])
        cv2.ellipse(frame,(cx,cy),(r,r),
                    -90,0,int(360*frac),arc_col,5,cv2.LINE_AA)
    _circle(frame,cx,cy,r-6,C["panel"])
    dur_str=f"{blink_dur:.1f}s" if blink_dur>0.02 else "OPEN"
    dur_col=(C["eye_bad"] if closed_valid else C["eye_ok"])
    ts2=cv2.getTextSize(dur_str,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)[0]
    _t(frame,dur_str,cx-ts2[0]//2,cy+5,0.42,dur_col)
    thresh_lbl="ALERT 0.5s  DROWSY 2.0s"
    _t(frame,thresh_lbl,px+4,cy+r+12,0.28,C["dim"])

    # ── Yawn counter ──────────────────────────────────────────────
    _t(frame,"YAWN COUNTER",px+8,cy+r+32,0.33,C["dim"])
    dot_y=cy+r+52; req=CFG.YAWN_DROWSY_COUNT
    dot_spacing=(pw-20)//(req+1)
    for i in range(req):
        dcx=px+20+i*dot_spacing+dot_spacing//2
        filled=i<yawn_count
        col=C["drowsy"] if yawn_drowsy else C["yawn"]
        _circle(frame,dcx,dot_y,9,col if filled else C["groove"])
        if filled:
            _circle(frame,dcx,dot_y,5,
                    tuple(min(255,v+80) for v in col))
        _circle(frame,dcx,dot_y,9,C["border"],fill=1)  # ring

    win_m=int(CFG.YAWN_DROWSY_WINDOW//60)
    _t(frame,f"{yawn_count}/{req} in {win_m}min",
       px+8,dot_y+20,0.36,
       C["drowsy"] if yawn_drowsy else C["silver"])

    # Active yawn bar
    if yawn_dur>0:
        by2=dot_y+32
        bw2=pw-16
        fill2=int(np.clip(yawn_dur/3.0,0,1)*bw2)
        _rounded(frame,px+8,by2,px+8+bw2,by2+7,3,C["groove"])
        if fill2>4:
            _rounded(frame,px+8,by2,px+8+fill2,by2+7,3,C["yawn"])
        _t(frame,f"Yawn {yawn_dur:.1f}s",px+8,by2-3,0.32,C["yawn"])

    # ── PERCLOS indicator ─────────────────────────────────────────
    ind_y=dot_y+60
    _t(frame,"PERCLOS",px+8,ind_y,0.33,C["dim"])
    pcl_col=(C["drowsy"] if perclos_val>=CFG.PERCLOS_DROWSY_THRESH
             else C["normal"])
    bw3=pw-16
    _rounded(frame,px+8,ind_y+8,px+8+bw3,ind_y+18,3,C["groove"])
    fill3=int(np.clip(perclos_val,0,1)*bw3)
    if fill3>4:
        _rounded(frame,px+8,ind_y+8,px+8+fill3,ind_y+18,3,pcl_col)
    tx3=px+8+int(CFG.PERCLOS_DROWSY_THRESH*bw3)
    _line(frame,(tx3,ind_y+5),(tx3,ind_y+21),C["alert"],2)
    _t(frame,f"{perclos_val*100:.1f}%",px+8+bw3+4,ind_y+16,0.38,pcl_col)

    # ── Active alert tags ─────────────────────────────────────────
    tag_y=ind_y+34
    if now<drowsy_until:
        _rounded(frame,px+8,tag_y,px+pw-8,tag_y+20,4,C["drowsy"])
        _t(frame,"DROWSY ALERT ACTIVE",px+14,tag_y+14,0.33,C["white"],1,True)
        tag_y+=26
    if now<yawn_until:
        _rounded(frame,px+8,tag_y,px+pw-8,tag_y+20,4,C["yawn"])
        _t(frame,"YAWN DETECTED",px+14,tag_y+14,0.33,C["panel"],1,True)


# ═══════════════════════════════════════════════════════════════════
#  VIDEO FRAME BORDER  (around the webcam feed)
# ═══════════════════════════════════════════════════════════════════
def draw_video_border(frame,vx,vy,vw,vh,status):
    col=Status.colour(status)
    # Corner brackets — automotive targeting style
    seg=22; thick=2
    corners=[
        (vx,vy,     1, 1),
        (vx+vw,vy,  -1, 1),
        (vx,vy+vh,  1,-1),
        (vx+vw,vy+vh,-1,-1),
    ]
    for cx,cy,sx,sy in corners:
        _line(frame,(cx,cy),(cx+sx*seg,cy),col,thick)
        _line(frame,(cx,cy),(cx,cy+sy*seg),col,thick)

    # Top label strip
    _blend(frame,vx,vy,vx+vw,vy+18,C["panel"],0.80)
    _t(frame,"LIVE FEED",vx+6,vy+13,0.32,C["dim"])
    ts=time.strftime("%Y-%m-%d")
    _t(frame,ts,vx+vw-80,vy+13,0.30,C["dim"])


# ═══════════════════════════════════════════════════════════════════
#  EYE PREVIEW THUMBNAILS
# ═══════════════════════════════════════════════════════════════════
def draw_eye_previews(frame,lc,rc,l_cl,r_cl,vx,vy,vw):
    tw,th=60,30
    for crop,closed,label,xoff in[
        (lc,l_cl,"L.EYE",vx+4),
        (rc,r_cl,"R.EYE",vx+vw-tw-4),
    ]:
        if crop is None or crop.size==0: continue
        try:
            g=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) if len(crop.shape)==3 else crop
            rsz=cv2.resize(g,(tw,th))
            col_img=cv2.cvtColor(rsz,cv2.COLOR_GRAY2BGR)
            # tint closed eye red
            if closed:
                overlay=col_img.copy()
                overlay[:,:,2]=np.minimum(255,overlay[:,:,2].astype(int)+80).astype(np.uint8)
                col_img=overlay
            by2=vy+22
            frame[by2:by2+th,xoff:xoff+tw]=col_img
            bc=C["eye_bad"] if closed else C["eye_ok"]
            _rounded(frame,xoff,by2,xoff+tw,by2+th,3,bc,fill=False)
            _t(frame,label,xoff+2,by2-3,0.28,bc)
        except: pass


# ═══════════════════════════════════════════════════════════════════
#  LANDMARK CONTOUR
# ═══════════════════════════════════════════════════════════════════
def draw_contour(frame,lm,idx,col,w,h):
    pts=[_px(lm,i,w,h).astype(int) for i in idx]
    for i in range(len(pts)):
        cv2.line(frame,tuple(pts[i]),tuple(pts[(i+1)%len(pts)]),
                 col,1,cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════
#  BOTTOM STATUS BAR
# ═══════════════════════════════════════════════════════════════════
def draw_bottom_bar(frame,status,ear_avg,mar_v,fw,fh):
    _blend(frame,0,fh-24,fw,fh,C["panel"],0.90)
    _line(frame,(0,fh-24),(fw,fh-24),C["border"],1)
    items=[
        (f"EAR {ear_avg:.3f}", C["eye_bad"] if ear_avg<CFG.EAR_THRESH else C["eye_ok"]),
        (f"MAR {mar_v:.3f}",   C["mth_bad"] if mar_v>CFG.MAR_THRESH   else C["mth_ok"]),
        (f"BLINK >{CFG.BLINK_MIN_SEC}s",  C["dim"]),
        (f"DROWSY >{CFG.DROWSY_MIN_SEC}s",C["dim"]),
        (f"YAWN x{CFG.YAWN_DROWSY_COUNT}/{int(CFG.YAWN_DROWSY_WINDOW//60)}min",C["dim"]),
        ("[M]esh  [D]ebug  [R]eset  [Q]uit", C["dim"]),
    ]
    x=8
    for txt,col in items:
        _t(frame,txt,x,fh-8,0.32,col)
        tw=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.32,1)[0][0]
        x+=tw+14
        if x>fw-180: break
    _t(frame,"[M]esh  [D]ebug  [R]eset  [Q]uit",fw-220,fh-8,0.30,C["dim"])


# ═══════════════════════════════════════════════════════════════════
#  DEBUG OVERLAY  (toggled with D)
# ═══════════════════════════════════════════════════════════════════
def draw_debug(frame,ear_l,ear_r,ear_avg,mar_v,
               ml_eye,ml_mouth,fused,
               blink_dur,closed_valid,
               yawn_dur,yawn_count,pclos,n_pc,fw,fh):
    vx=176; vw=fw-176-176
    px=vx+4; py=56; lh=18
    _blend(frame,px-2,py,px+210,py+lh*12+4,C["void"],0.78)
    rows=[
        ("EAR L/R/avg",f"{ear_l:.3f} / {ear_r:.3f} / {ear_avg:.3f}",
         C["eye_bad"] if ear_avg<CFG.EAR_THRESH else C["eye_ok"]),
        ("MAR",        f"{mar_v:.3f}",
         C["mth_bad"] if mar_v>CFG.MAR_THRESH else C["mth_ok"]),
        ("ML-Eye",     f"{ml_eye:.3f}",
         C["alert"] if ml_eye>=CFG.ML_EYE_THRESH else C["silver"]),
        ("ML-Mouth",   f"{ml_mouth:.3f}",
         C["alert"] if ml_mouth>=CFG.ML_MOUTH_THRESH else C["silver"]),
        ("Fused",      f"{fused:.3f}",  C["cyan"]),
        ("Blink-t",    f"{blink_dur:.2f}s {'[VALID]' if closed_valid else ''}",
         C["alert"] if closed_valid else C["dim"]),
        ("Yawn-t/cnt", f"{yawn_dur:.2f}s / {yawn_count}",C["yawn"]),
        ("PERCLOS",    f"{pclos*100:.1f}%  ({n_pc} frames)",
         C["drowsy"] if pclos>=CFG.PERCLOS_DROWSY_THRESH else C["normal"]),
    ]
    for i,(lbl,val,col) in enumerate(rows):
        _t(frame,f"{lbl:<16}{val}",px+2,py+10+i*lh,0.36,col)


# ═══════════════════════════════════════════════════════════════════
#  CSV
# ═══════════════════════════════════════════════════════════════════
def init_log(p):
    p.parent.mkdir(parents=True,exist_ok=True)
    f=open(p,"w",newline="",encoding="utf-8")
    w=csv.writer(f)
    w.writerow(["timestamp","frame","fps","ear_left","ear_right",
                "ear_avg","mar","ml_eye_conf","ml_mouth_conf",
                "fused","eye_closed_raw","eye_closed_valid",
                "blink_dur","yawn_dur","yawn_count","perclos","status"])
    return f,w


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    csv_file,csv_w=init_log(CFG.LOG_FILE)

    print("\n"+"═"*56)
    print("  DRIVEWATCH — Driver Drowsiness Monitor  Loading")
    print("═"*56)

    try:
        engine=DrowsinessInferenceEngine(
            eye_model_path=CFG.EYE_ONNX,
            mouth_model_path=CFG.MOUTH_ONNX,verbose=True)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}"); sys.exit(1)

    engine.warmup(10); print("Engine ready.\n")

    mp_fm=mp.solutions.face_mesh
    face_mesh=mp_fm.FaceMesh(static_image_mode=False,max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=CFG.MIN_DETECT,
        min_tracking_confidence=CFG.MIN_TRACK)

    cap=cv2.VideoCapture(CFG.CAM_INDEX)
    if not cap.isOpened(): print("[ERROR] Camera not found"); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,CFG.CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CFG.CAM_H)

    # State
    blink=BlinkFilter(); yawn_ctr=YawnCounter()
    perclos=Perclos(); snd=Sound()
    drowsy_until=yawn_until=0.0
    show_debug=False; show_mesh=False
    flash_on=False; flash_ctr=0
    fps=0.0; tick=cv2.getTickCount(); fidx=0

    # Working vars
    ear_l=ear_r=ear_avg=mar_v=0.0
    ml_eye=ml_mouth=fused=0.0
    eye_raw=eye_valid=mouth_fused=False
    lc=rc=None; l_cl=r_cl=False
    status=Status.NORMAL

    # Canvas dimensions (larger window with panels)
    FW,FH=980,560
    # Video area inside panels
    VX,VY=176,54; VW=FW-176-176-2; VH=FH-54-24-2

    _build_scanlines(FH,FW)

    print("Running.  [Q]quit  [M]mesh  [D]debug  [R]reset\n")

    while True:
        ret,cam_frame=cap.read()
        if not ret: continue

        cam_frame=cv2.flip(cam_frame,1)
        ch,cw=cam_frame.shape[:2]
        now=time.time()

        # ── MediaPipe on cam_frame ────────────────────────────────
        rgb=cv2.cvtColor(cam_frame,cv2.COLOR_BGR2RGB)
        rgb.flags.writeable=False
        results=face_mesh.process(rgb)
        rgb.flags.writeable=True
        face_found=bool(results.multi_face_landmarks)

        if face_found:
            lm=results.multi_face_landmarks[0].landmark
            ear_l=ear_(lm,_L_EAR,cw,ch)
            ear_r=ear_(lm,_R_EAR,cw,ch)
            ear_avg=(ear_l+ear_r)/2.0
            mar_v=mar_(lm,_M_MAR,cw,ch)

            lc=crop_(cam_frame,lm,_L_RING,cw,ch,0.18)
            rc=crop_(cam_frame,lm,_R_RING,cw,ch,0.18)
            mc=crop_(cam_frame,lm,_M_CROP,cw,ch,0.20)

            ml_eye=0.0; l_cl=r_cl=False
            if lc is not None and rc is not None:
                lr,rr=engine.predict_eyes_batch(lc,rc)
                ml_eye=(lr.probabilities[0]+rr.probabilities[0])/2
                l_cl=lr.closed; r_cl=rr.closed
            elif lc is not None:
                res=engine.predict_eye(lc)
                ml_eye=res.probabilities[0]; l_cl=res.closed

            ml_mouth=0.0
            if mc is not None:
                mr=engine.predict_mouth(mc)
                ml_mouth=mr.probabilities[1]

            fused,eye_raw=fuse_eye(ear_avg,ml_eye)
            mouth_fused=fuse_mouth(mar_v,ml_mouth)

            blink.update(eye_raw); eye_valid=blink.closed_valid
            yawn_ctr.update(mouth_fused)
            perclos.update(eye_valid)
            pclos_v=perclos.value

            if (blink.duration>=CFG.DROWSY_MIN_SEC or
                    pclos_v>=CFG.PERCLOS_DROWSY_THRESH or
                    yawn_ctr.drowsy):
                drowsy_until=now+CFG.ALERT_LINGER
            if yawn_ctr.just_confirmed:
                yawn_until=now+CFG.ALERT_LINGER

            if now<drowsy_until: status=Status.DROWSY; snd.beep()
            elif now<yawn_until: status=Status.YAWN
            elif eye_valid:      status=Status.ALERT
            else:                status=Status.NORMAL

            e_col=C["eye_bad"] if eye_valid else C["eye_ok"]
            m_col=C["mth_bad"] if mouth_fused else C["mth_ok"]
            if show_mesh:
                for conn in mp_fm.FACEMESH_TESSELATION:
                    p1,p2=lm[conn[0]],lm[conn[1]]
                    cv2.line(cam_frame,
                             (int(p1.x*cw),int(p1.y*ch)),
                             (int(p2.x*cw),int(p2.y*ch)),
                             C["mesh"],1,cv2.LINE_AA)
            draw_contour(cam_frame,lm,_L_RING,e_col,cw,ch)
            draw_contour(cam_frame,lm,_R_RING,e_col,cw,ch)
            draw_contour(cam_frame,lm,_M_CROP,m_col,cw,ch)
        else:
            blink.reset(); ear_l=ear_r=ear_avg=mar_v=0.0
            ml_eye=ml_mouth=fused=0.0
            eye_raw=eye_valid=mouth_fused=False
            perclos.update(False); pclos_v=perclos.value
            status=Status.NORMAL
            cv2.putText(cam_frame,"NO FACE DETECTED",
                (cw//2-120,ch//2),cv2.FONT_HERSHEY_DUPLEX,
                0.8,C["alert"],2,cv2.LINE_AA)

        # ── Build canvas ──────────────────────────────────────────
        canvas=np.full((FH,FW,3),C["void"],dtype=np.uint8)

        # Paste video feed
        vid=cv2.resize(cam_frame,(VW,VH))
        canvas[VY:VY+VH, VX:VX+VW]=vid

        # Panels
        draw_header(canvas,fps,FW,FH)
        draw_left_panel(canvas,ear_avg,mar_v,ml_eye,ml_mouth,
                        fused,pclos_v,FW,FH)
        draw_right_panel(canvas,blink.duration,eye_valid,
                         yawn_ctr.count,yawn_ctr.cur_dur,
                         yawn_ctr.drowsy,drowsy_until,yawn_until,
                         pclos_v,FW,FH)
        draw_video_border(canvas,VX,VY,VW,VH,status)
        draw_eye_previews(canvas,lc,rc,l_cl,r_cl,VX,VY,VW)
        draw_bottom_bar(canvas,status,ear_avg,mar_v,FW,FH)

        # Status pill
        flash_ctr+=1
        if flash_ctr>=CFG.FLASH_EVERY: flash_on=not flash_on; flash_ctr=0
        draw_status_pill(canvas,status,flash_on,FW,FH)

        if show_debug:
            draw_debug(canvas,ear_l,ear_r,ear_avg,mar_v,
                       ml_eye,ml_mouth,fused,
                       blink.duration,eye_valid,
                       yawn_ctr.cur_dur,yawn_ctr.count,
                       pclos_v,perclos.n,FW,FH)

        # Scanline texture
        apply_scanlines(canvas)

        # FPS
        nt=cv2.getTickCount()
        fps=cv2.getTickFrequency()/(nt-tick+1e-9); tick=nt

        # CSV
        csv_w.writerow([f"{now:.4f}",fidx,f"{fps:.1f}",
            f"{ear_l:.4f}",f"{ear_r:.4f}",f"{ear_avg:.4f}",
            f"{mar_v:.4f}",f"{ml_eye:.4f}",f"{ml_mouth:.4f}",
            f"{fused:.4f}",int(eye_raw),int(eye_valid),
            f"{blink.duration:.4f}",f"{yawn_ctr.cur_dur:.4f}",
            yawn_ctr.count,f"{pclos_v:.4f}",status])
        fidx+=1

        cv2.imshow(CFG.WIN_TITLE,canvas)
        key=cv2.waitKey(1)&0xFF
        if   key==ord('q'): break
        elif key==ord('m'): show_mesh=not show_mesh
        elif key==ord('d'): show_debug=not show_debug
        elif key==ord('r'):
            perclos.reset(); yawn_ctr.reset(); blink.reset()
            print("[RESET] All counters cleared.")

    cap.release(); csv_file.close()
    cv2.destroyAllWindows(); face_mesh.close()
    print(f"\nSession ended — {fidx} frames.  Log → {CFG.LOG_FILE.resolve()}")


if __name__=="__main__":
    main()