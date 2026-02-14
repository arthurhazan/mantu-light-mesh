import { useState, useRef, useCallback, useEffect, useMemo } from "react";

/* ═══════════════════════════════════════════════════════════════════════════
   LIGHT MESH · WebGL Mesh Gradient Generator
   ═══════════════════════════════════════════════════════════════════════════ */

const PRACTICES = {
  mind: { hex: "#D4B8FF", rgb: [0.831, 0.722, 1.0], label: "Mind" },
  core: { hex: "#B0F0FF", rgb: [0.690, 0.941, 1.0], label: "Core" },
  soma: { hex: "#D4FFD4", rgb: [0.831, 1.0, 0.831], label: "Soma" },
  aura: { hex: "#FFD6FF", rgb: [1.0, 0.839, 1.0], label: "Aura" },
};
const P_KEYS = ["mind", "core", "soma", "aura"];
const MAX_PTS = 16;
const SHAPES = [{ id: 0, l: "●", n: "Point" }, { id: 1, l: "○", n: "Circle" }, { id: 2, l: "□", n: "Square" }, { id: 3, l: "◇", n: "Diamond" }, { id: 4, l: "―", n: "Line" }, { id: 5, l: "M", n: "Mantu" }];

/* ─── Shaders ───────────────────────────────────────────────────────────── */
const VS = `attribute vec2 a_pos;void main(){gl_Position=vec4(a_pos,0,1);}`;
const FS = `
precision highp float;
uniform vec2 u_res;
uniform vec2 u_pt[${MAX_PTS}];
uniform vec3 u_cl[${MAX_PTS}];
uniform float u_rad[${MAX_PTS}],u_int[${MAX_PTS}],u_sft[${MAX_PTS}],u_str[${MAX_PTS}],u_ang[${MAX_PTS}];
uniform float u_shp[${MAX_PTS}],u_sz[${MAX_PTS}];
uniform int u_n;
uniform float u_pw,u_t;
uniform float u_contrast,u_bloom,u_grain,u_chroma,u_vig,u_scanF,u_scanI;
uniform sampler2D u_tex;

float h(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float ns(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
  return mix(mix(h(i),h(i+vec2(1,0)),f.x),mix(h(i+vec2(0,1)),h(i+vec2(1,1)),f.x),f.y);}

float sdSeg(vec2 p,vec2 a,vec2 b){vec2 pa=p-a,ba=b-a;return length(pa-ba*clamp(dot(pa,ba)/dot(ba,ba),0.,1.));}
float sdBox(vec2 p,vec2 b){vec2 d=abs(p)-b;return length(max(d,vec2(0.)))+min(max(d.x,d.y),0.);}

float sDist(vec2 uv,float shp,vec2 ctr,float sz,float ang,float str){
  vec2 d=uv-ctr;float ca=cos(ang),sa=sin(ang);
  d=vec2(ca*d.x+sa*d.y,-sa*d.x+ca*d.y);d.x/=max(str,.01);
  
  if(shp>.5&&shp<5.5) d.x*=str; // Correct scaling only for texture/non-simple
  if(shp<.5)return length(d); // Point
  if(shp<1.5)return max(length(d)-sz, 0.); // Circle
  if(shp<2.5)return max(sdBox(d,vec2(sz)), 0.); // Square
  if(shp<3.5){vec2 r=vec2(d.x+d.y,d.y-d.x)*.7071;return max(sdBox(r,vec2(sz*.7)), 0.);} // Diamond
  if(shp<4.5)return sdSeg(d,vec2(-sz,0.),vec2(sz,0.)); // Line
  
  vec2 box=vec2(sz*3.0,sz);
  vec2 tUV=d/box/2.+.5;
  float bD=sdBox(d,box);
  if(bD>0.) return texture2D(u_tex,clamp(tUV,0.,1.)).a+bD;
  return texture2D(u_tex,tUV).a;
}

vec3 grad(vec2 uv){
  vec3 c=vec3(0.);float tw=0.;
  for(int i=0;i<${MAX_PTS};i++){if(i>=u_n)break;
    float dist=sDist(uv,u_shp[i],u_pt[i],u_sz[i],u_ang[i],u_str[i])/max(u_rad[i],.01);
    float w=u_int[i]/pow(max(dist,.0001),u_pw*u_sft[i]);
    c+=u_cl[i]*w;tw+=w;}
  return c/tw;}
void main(){
  vec2 uv=gl_FragCoord.xy/u_res;uv.y=1.-uv.y;
  float n=ns(uv*5.+u_t*.03)*.012;uv+=vec2(n,n*.8);
  vec3 col;
  if(u_chroma>.001){vec2 dir=(uv-.5)*u_chroma;
    col=vec3(grad(uv+dir).r,grad(uv).g,grad(uv-dir).b);}
  else col=grad(uv);
  col=pow(max(col,vec3(0.)),vec3(u_contrast));
  float lum=dot(col,vec3(.2126,.7152,.0722));
  col+=col*smoothstep(.4,1.,lum)*u_bloom;
  float vig=smoothstep(1.2,.3,length((uv-.5)*1.8)*u_vig);
  col*=mix(.85,1.,vig);
  if(u_scanI>.001){float sc=smoothstep(.3,.5,abs(sin(uv.x*u_scanF*100.)));
    col*=mix(1.,sc,u_scanI);}
  col+=vec3((h(uv*u_res*.01+u_t)-.5)*u_grain);
  gl_FragColor=vec4(clamp(col,0.,1.),1.);}`;

/* ─── WebGL helpers ─────────────────────────────────────────────────────── */
function mkSh(gl, t, s) {
  const sh = gl.createShader(t); gl.shaderSource(sh, s); gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) { console.error(gl.getShaderInfoLog(sh)); return null; } return sh;
}
function mkPg(gl) {
  const vs = mkSh(gl, gl.VERTEX_SHADER, VS), fs = mkSh(gl, gl.FRAGMENT_SHADER, FS);
  if (!vs || !fs) return null; const p = gl.createProgram(); gl.attachShader(p, vs); gl.attachShader(p, fs); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) { console.error(gl.getProgramInfoLog(p)); return null; } return p;
}

/* ─── CSS Export ────────────────────────────────────────────────────────── */
function buildCSS(pts) {
  const col = pts.filter(p => p.ck !== "black");
  if (!col.length) return "/* Add a practice color to at least one point */";
  let s = `/* ─── Mesh Gradient (CSS approximation) ─── */\n`;
  s += `.mesh-gradient {\n  position: relative;\n  width: 100%; height: 100vh;\n  background: #000;\n  overflow: hidden;\n}\n\n`;
  col.forEach((pt, i) => {
    const p = PRACTICES[pt.ck]; if (!p) return;
    const [r, g, b] = p.rgb.map(v => Math.round(v * 255));
    const rad = Math.round((pt.radius || .5) * 140);
    s += `.mesh-gradient__glow-${i + 1} {\n  position: absolute;\n`;
    s += `  left: ${(pt.x * 100).toFixed(1)}%;\n  top: ${(pt.y * 100).toFixed(1)}%;\n`;
    s += `  width: ${rad}%; height: ${rad}%;\n  transform: translate(-50%,-50%)${pt.angle ? ` rotate(${pt.angle}deg)` : ""};\n`;
    s += `  background: radial-gradient(${pt.stretch !== 1 ? `${Math.round(50 / pt.stretch)}% 50%` : "50% 50%"} at 50% 50%,\n`;
    s += `    rgba(${r},${g},${b},${(0.9 * (pt.intensity || 1)).toFixed(2)}) 0%,\n`;
    s += `    rgba(${r},${g},${b},0.08) 65%, transparent 100%);\n`;
    s += `  mix-blend-mode: screen;\n  filter: blur(${Math.round(50 / (pt.softness || 1))}px);\n  pointer-events: none;\n}\n\n`;
  });
  s += `/* HTML */\n/*\n<div class="mesh-gradient">\n`;
  col.forEach((_, i) => { s += `  <div class="mesh-gradient__glow-${i + 1}"></div>\n`; });
  s += `</div>\n*/`; return s;
}

/* ─── Default points ────────────────────────────────────────────────────── */
let _id = 0; function nid() { return _id++; }
const DP = { radius: .5, intensity: 1, softness: 1, stretch: 1, angle: 0, shape: 0, shapeSize: .15 };
function defaults() {
  return [
    { x: .08, y: .08, ck: "black", id: nid(), ...DP },
    { x: .92, y: .08, ck: "black", id: nid(), ...DP },
    { x: .08, y: .92, ck: "black", id: nid(), ...DP },
    { x: .92, y: .92, ck: "black", id: nid(), ...DP },
    { x: .50, y: .42, ck: "mind", id: nid(), ...DP },
    { x: .32, y: .68, ck: "black", id: nid(), ...DP },
    { x: .68, y: .28, ck: "black", id: nid(), ...DP },
  ];
}

/* ─── Styles ────────────────────────────────────────────────────────────── */
const btnS = (a) => ({
  padding: "6px 14px", background: a ? "rgba(255,255,255,.12)" : "rgba(255,255,255,.06)",
  border: "1px solid rgba(255,255,255,.1)", borderRadius: 6, color: "#fff", fontSize: 11,
  fontFamily: "inherit", cursor: "pointer", transition: "all .15s", display: "inline-flex", alignItems: "center", gap: 6
});
const panelS = {
  position: "absolute", width: 195, background: "rgba(8,8,12,.88)",
  border: "1px solid rgba(255,255,255,.08)", borderRadius: 10, zIndex: 30,
  backdropFilter: "blur(16px)", WebkitBackdropFilter: "blur(16px)", padding: "10px 12px"
};
const panelHead = { fontSize: 9, color: "rgba(255,255,255,.4)", letterSpacing: ".06em", marginBottom: 8, textTransform: "uppercase" };

/* ─── Slider component ─────────────────────────────────────────────────── */
function Sl({ l, v, set, mn, mx, st }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 4 }}>
      <span style={{ fontSize: 8, color: "rgba(255,255,255,.4)", width: 52, flexShrink: 0 }}>{l}</span>
      <input type="range" min={mn} max={mx} step={st || .1} value={v} onChange={e => set(+e.target.value)}
        style={{ flex: 1, accentColor: "#D4B8FF", height: 2 }} />
      <span style={{ fontSize: 8, color: "rgba(255,255,255,.3)", width: 26, textAlign: "right" }}>{v.toFixed(st && st < .1 ? 2 : 1)}</span>
    </div>
  );
}

/* ─── SDF Generator ─── */
function genSDF() {
  const W = 2048, H = 680;
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const x = c.getContext("2d");
  x.fillStyle = "#000"; x.fillRect(0, 0, W, H);

  const p = [
    "M352.404,197.609c-4.391,9.829-15.058,16.94-28.86,16.94-30.323,0-50.4-23.631-50.4-54.792,0-30.115,19.658-53.956,49.564-53.956,18.403,0,26.978,9.829,29.487,15.685v-7.861c0-2.704,2.192-4.896,4.896-4.896h17.186c2.704,0,4.896,2.192,4.896,4.896v79.382c0,5.357.213,10.005.432,13.407.175,2.694-1.868,5.021-4.562,5.196-.107.007-.215.01-.323.01h-16.99c-2.516-.002-4.62-1.911-4.868-4.414-.253-2.476-.458-5.477-.458-8.342v-1.255ZM326.89,190.499c14.43,0,25.305-11.922,25.305-30.743s-10.665-29.906-25.305-29.906c-14.848,0-25.723,11.084-25.723,29.906,0,18.611,10.665,30.743,25.723,30.743h0Z",
    "M419.633,211.62h-18.022c-2.704,0-4.896-2.192-4.896-4.896h0v-93.099c0-2.704,2.192-4.896,4.896-4.896h17.186c2.704,0,4.896,2.192,4.896,4.896h0v7.861c6.274-10.665,18.612-15.476,29.696-15.476,25.515,0,37.226,18.195,37.226,40.78v59.934c0,2.704-2.192,4.896-4.896,4.896h-18.022c-2.704,0-4.896-2.192-4.896-4.896h0v-55.125c0-11.502-5.646-20.494-19.031-20.494-12.129,0-19.24,9.41-19.24,21.331v54.288c0,2.704-2.192,4.896-4.896,4.896Z",
    "M648.755,200.537c-5.646,9.619-17.567,13.803-28.442,13.803-24.886,0-39.107-18.195-39.107-40.571v-60.143c0-2.704,2.192-4.896,4.896-4.896h18.022c2.704,0,4.896,2.192,4.896,4.896h0v54.496c0,11.502,5.856,20.705,18.822,20.705,12.339,0,19.449-8.366,19.449-20.286v-54.914c0-2.704,2.192-4.896,4.896-4.896h18.022c2.704,0,4.896,2.192,4.896,4.896h0v79.382c0,5.01.249,9.686.538,13.35.208,2.691-1.805,5.041-4.497,5.249-.128.01-.256.015-.385.014h-16.844c-2.543.005-4.665-1.942-4.879-4.476-.169-2.054-.284-4.417-.284-6.608Z",
    "M569.392,113.625v17.161c0,2.704-2.192,4.896-4.896,4.896h-18.034v40.805c0,8.992,4.182,11.92,12.129,11.92,2.88.028,5.754-.253,8.575-.837v19.5c-.001,2.225-1.518,4.162-3.677,4.699-3.934.929-7.967,1.371-12.008,1.315-.186,0-.573-.001-.757-.003-19.753-.218-32.069-12.101-32.069-31.785v-45.614h-15.966c-2.704,0-4.896-2.192-4.896-4.896h0v-17.161c0-2.704,2.192-4.896,4.896-4.896h61.808c2.704,0,4.896,2.192,4.896,4.896h0Z",
    "M257.346,214.494c-5.816,0-11.264-3.616-13.322-9.41l-17.211-48.464-23.963,49.856c-2.354,4.898-7.308,8.013-12.742,8.013s-10.387-3.115-12.742-8.013l-23.963-49.856-17.211,48.464c-2.614,7.357-10.696,11.206-18.053,8.59-7.357-2.612-11.204-10.695-8.591-18.053l28.57-80.448c1.91-5.379,6.868-9.081,12.567-9.386,5.705-.311,11.024,2.848,13.497,7.993l25.926,53.939,25.926-53.939c2.473-5.145,7.792-8.306,13.497-7.993,5.699.305,10.657,4.007,12.567,9.386l28.57,80.448c2.613,7.357-1.234,15.441-8.591,18.053-1.563.556-3.161.819-4.731.819Z"
  ];
  x.translate(240, 20); x.scale(2.0, 2.0);
  x.fillStyle = "#fff";
  p.forEach(s => x.fill(new Path2D(s)));

  const id = x.getImageData(0, 0, W, H); const d = id.data;
  const dist = new Float32Array(W * H); const inf = 1e4;

  // Init dist: 0 inside (white), inf outside
  for (let i = 0; i < W * H; i++) dist[i] = d[i * 4] > 128 ? 0 : inf;

  // BFS Distance Transform
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      const idx = i * W + j;
      if (dist[idx] > 0) {
        if (i > 0) dist[idx] = Math.min(dist[idx], dist[((i - 1) * W) + j] + 1);
        if (j > 0) dist[idx] = Math.min(dist[idx], dist[idx - 1] + 1);
      }
    }
  }
  for (let i = H - 1; i >= 0; i--) {
    for (let j = W - 1; j >= 0; j--) {
      const idx = i * W + j;
      if (dist[idx] > 0) {
        if (i < H - 1) dist[idx] = Math.min(dist[idx], dist[((i + 1) * W) + j] + 1);
        if (j < W - 1) dist[idx] = Math.min(dist[idx], dist[idx + 1] + 1);
      }
    }
  }

  // Convert to Uint8 for texture, scaled for 2048px width
  const tex = new Uint8Array(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    const v = dist[i];
    tex[i * 4] = 255; tex[i * 4 + 1] = 255; tex[i * 4 + 2] = 255;
    tex[i * 4 + 3] = Math.min(v * 0.8, 255);
  }
  return { d: tex, w: W, h: H };
}

/* ═══════════════════════════════════════════════════════════════════════════ */
export default function LightDiffuser() {
  const [pts, setPts] = useState(defaults);
  const [sel, setSel] = useState(null);
  const [active, setAct] = useState(["mind"]);
  const [cssVis, setCss] = useState(false);
  const [power, setPow] = useState(3.2);
  const [drag, setDrag] = useState(null);
  const [showFil, setShowFil] = useState(false);
  const [fil, setFil] = useState({ contrast: 1, bloom: 0, grain: 0, chroma: 0, vignette: 1, scanFreq: 50, scanInt: 0 });
  const [tutStep, setTutStep] = useState(-1);

  const cvRef = useRef(null), boxRef = useRef(null);
  const glR = useRef(null), pgR = useRef(null), afR = useRef(null), texR = useRef(null);
  const t0 = useRef(Date.now()), didI = useRef(false);

  const updPt = useCallback((i, k, v) => { setPts(p => p.map((pt, j) => j === i ? { ...pt, [k]: v } : pt)); }, []);
  const updFil = useCallback((k, v) => { setFil(p => ({ ...p, [k]: v })); }, []);

  /* ── WebGL init ── */
  useEffect(() => {
    const c = cvRef.current; if (!c) return;
    const dpr = devicePixelRatio || 1, r = c.getBoundingClientRect(); c.width = r.width * dpr; c.height = r.height * dpr;

    // Gen Texture SDF
    const sdf = genSDF();

    const gl = c.getContext("webgl", { preserveDrawingBuffer: true, antialias: true }); if (!gl) return;
    const pg = mkPg(gl); if (!pg) return;

    // Upload Texture
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, sdf.w, sdf.h, 0, gl.RGBA, gl.UNSIGNED_BYTE, sdf.d);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    texR.current = tex;

    const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(pg, "a_pos"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    glR.current = gl; pgR.current = pg;
    return () => { gl.deleteProgram(pg); gl.deleteBuffer(buf); gl.deleteTexture(tex); };
  }, []);


  /* ── Resize ── */
  useEffect(() => {
    const fn = () => {
      const c = cvRef.current; if (!c) return;
      const d = devicePixelRatio || 1, r = c.getBoundingClientRect(); c.width = r.width * d; c.height = r.height * d;
    };
    window.addEventListener("resize", fn); return () => window.removeEventListener("resize", fn);
  }, []);

  /* ── Tutorial ── */
  useEffect(() => {
    if (!localStorage.getItem("mantu_tut_done")) setTutStep(0);
  }, []);
  const nextTut = () => {
    if (tutStep >= 3) {
      localStorage.setItem("mantu_tut_done", "true");
      setTutStep(-1);
    } else {
      setTutStep(s => s + 1);
    }
  };
  const skipTut = () => {
    localStorage.setItem("mantu_tut_done", "true");
    setTutStep(-1);
  };

  /* ── Render loop ── */
  useEffect(() => {
    const gl = glR.current, pg = pgR.current; if (!gl || !pg) return;
    const uloc = (n) => gl.getUniformLocation(pg, n);
    const frame = () => {
      const c = cvRef.current; if (!c) return;
      gl.viewport(0, 0, c.width, c.height); gl.useProgram(pg);
      gl.uniform2f(uloc("u_res"), c.width, c.height);
      gl.uniform1f(uloc("u_t"), (Date.now() - t0.current) / 1000);
      gl.uniform1f(uloc("u_pw"), power);
      gl.uniform1i(uloc("u_n"), pts.length);
      for (let i = 0; i < MAX_PTS; i++) {
        const p = pts[i]; if (p) {
          gl.uniform2f(uloc(`u_pt[${i}]`), p.x, p.y);
          const rgb = p.ck === "black" ? [0, 0, 0] : (PRACTICES[p.ck]?.rgb || [0, 0, 0]);
          gl.uniform3f(uloc(`u_cl[${i}]`), rgb[0], rgb[1], rgb[2]);
          gl.uniform1f(uloc(`u_rad[${i}]`), p.radius || .5);
          gl.uniform1f(uloc(`u_int[${i}]`), p.intensity || 1);
          gl.uniform1f(uloc(`u_sft[${i}]`), p.softness || 1);
          gl.uniform1f(uloc(`u_str[${i}]`), p.stretch || 1);
          gl.uniform1f(uloc(`u_ang[${i}]`), (p.angle || 0) * Math.PI / 180);
          gl.uniform1f(uloc(`u_shp[${i}]`), p.shape || 0);
          gl.uniform1f(uloc(`u_sz[${i}]`), p.shapeSize || .15);
        }
      }
      gl.uniform1f(uloc("u_contrast"), fil.contrast);
      gl.uniform1f(uloc("u_bloom"), fil.bloom);
      gl.uniform1f(uloc("u_grain"), fil.grain);
      gl.uniform1f(uloc("u_chroma"), fil.chroma);
      gl.uniform1f(uloc("u_vig"), fil.vignette);
      gl.uniform1f(uloc("u_scanF"), fil.scanFreq);
      gl.uniform1f(uloc("u_scanI"), fil.scanInt);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texR.current);
      gl.uniform1i(uloc("u_tex"), 0);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      afR.current = requestAnimationFrame(frame);
    };
    frame(); return () => { if (afR.current) cancelAnimationFrame(afR.current); };
  }, [pts, power, fil]);

  /* ── Pointer ── */
  const gp = useCallback((e) => {
    const r = boxRef.current?.getBoundingClientRect(); if (!r) return { x: .5, y: .5 };
    const cx = e.touches ? e.touches[0].clientX : e.clientX, cy = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: Math.max(0, Math.min(1, (cx - r.left) / r.width)), y: Math.max(0, Math.min(1, (cy - r.top) / r.height)) };
  }, []);
  const hDown = useCallback((e, i) => { e.stopPropagation(); e.preventDefault(); didI.current = true; setDrag(i); setSel(i); }, []);
  const hMove = useCallback((e) => {
    if (drag === null) return; e.preventDefault(); const p = gp(e);
    setPts(prev => prev.map((pt, i) => i === drag ? { ...pt, x: p.x, y: p.y } : pt));
  }, [drag, gp]);
  const hUp = useCallback(() => { if (drag !== null) didI.current = true; setDrag(null); }, [drag]);

  useEffect(() => {
    if (drag !== null) {
      window.addEventListener("mousemove", hMove); window.addEventListener("mouseup", hUp);
      window.addEventListener("touchmove", hMove, { passive: false }); window.addEventListener("touchend", hUp);
      return () => {
        window.removeEventListener("mousemove", hMove); window.removeEventListener("mouseup", hUp);
        window.removeEventListener("touchmove", hMove); window.removeEventListener("touchend", hUp);
      };
    }
  }, [drag, hMove, hUp]);

  const hClick = useCallback((e) => {
    if (didI.current) { didI.current = false; return; }
    if (pts.length >= MAX_PTS) return; const p = gp(e);
    setPts(prev => [...prev, { x: p.x, y: p.y, ck: "black", id: nid(), ...DP }]); setSel(pts.length);
  }, [gp, pts.length]);

  /* ── Actions ── */
  const assign = useCallback((ck) => {
    if (sel === null) return;
    setPts(prev => prev.map((p, i) => i === sel ? { ...p, ck } : p));
  }, [sel]);
  const remove = useCallback(() => {
    if (sel === null || pts.length <= 3) return;
    setPts(prev => prev.filter((_, i) => i !== sel)); setSel(null);
  }, [sel, pts.length]);
  const clearAll = useCallback(() => { setPts([]); setSel(null); }, []);
  const toggleAct = useCallback((k) => {
    setAct(p => p.includes(k) ? (p.length > 1 ? p.filter(c => c !== k) : p) : [...p, k]);
  }, []);

  const shuffle = useCallback(() => {
    setPts(prev => {
      const n = prev.map(p => ({ ...p, x: .06 + Math.random() * .88, y: .06 + Math.random() * .88, ck: "black" }));
      const idx = n.map((_, i) => i).sort(() => Math.random() - .5);
      active.forEach((ck, ci) => {
        if (ci < idx.length) n[idx[ci]].ck = ck;
        if (ci + active.length < idx.length && Math.random() > .4) n[idx[ci + active.length]].ck = ck;
      });
      return n;
    }); setSel(null);
  }, [active]);

  const dlPNG = useCallback(() => {
    const c = cvRef.current; if (!c) return;
    const a = document.createElement("a"); a.download = "light-mesh.png"; a.href = c.toDataURL("image/png"); a.click();
  }, []);
  const css = useMemo(() => buildCSS(pts), [pts]);
  const cpCSS = useCallback(() => navigator.clipboard.writeText(css), [css]);

  /* ── Keyboard ── */
  useEffect(() => {
    const fn = (e) => {
      if ((e.key === "Delete" || e.key === "Backspace") && sel !== null && pts.length > 3) {
        e.preventDefault(); remove();
      }
    };
    window.addEventListener("keydown", fn); return () => window.removeEventListener("keydown", fn);
  }, [sel, pts.length, remove]);

  const selData = sel !== null ? pts[sel] : null;

  /* ═══ RENDER ═══ */
  return (
    <div ref={boxRef} onClick={hClick} style={{
      position: "relative", width: "100vw", height: "100vh",
      background: "#000", overflow: "hidden", cursor: "crosshair",
      fontFamily: "'IBM Plex Mono','SF Mono',monospace", touchAction: "none"
    }}>
      <canvas ref={cvRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} />

      {/* ── Handles ── */}
      {pts.map((pt, i) => {
        const c = pt.ck === "black" ? "#333" : PRACTICES[pt.ck].hex; const isSel = sel === i;
        return (<div key={pt.id} onMouseDown={(e) => hDown(e, i)} onTouchStart={(e) => hDown(e, i)}
          onDoubleClick={(e) => {
            e.stopPropagation(); const keys = ["black", ...P_KEYS];
            assign(keys[(keys.indexOf(pt.ck) + 1) % keys.length]);
          }}
          style={{
            position: "absolute", left: `${pt.x * 100}%`, top: `${pt.y * 100}%`,
            transform: "translate(-50%,-50%)", width: 16, height: 16, borderRadius: "50%",
            background: c, border: `2px solid ${isSel ? "#fff" : "rgba(255,255,255,.3)"}`,
            cursor: "grab", zIndex: 10, transition: "border-color .15s, box-shadow .15s",
            boxShadow: isSel ? `0 0 16px ${c}88, 0 0 4px #fff4` : `0 0 8px ${c}44`
          }}>
          <div style={{
            position: "absolute", top: -20, left: "50%", transform: "translateX(-50%)",
            fontSize: 8, color: "rgba(255,255,255,.55)", whiteSpace: "nowrap", pointerEvents: "none"
          }}>
            {pt.ck === "black" ? "" : PRACTICES[pt.ck].label}
          </div></div>);
      })}

      {/* Empty state */}
      {pts.filter(p => p.ck !== "black").length === 0 && (
        <div style={{
          position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
          color: "rgba(255,255,255,.2)", fontSize: 14, pointerEvents: "none", flexDirection: "column", gap: 6
        }}>
          <span style={{ fontSize: 28, opacity: .3 }}>✦</span>Select a point and assign a color below
        </div>)}

      {/* ═══ TOP TOOLBAR ═══ */}
      <div style={{
        position: "absolute", top: 0, left: 0, right: 0, padding: "14px 20px", display: "flex",
        alignItems: "center", justifyContent: "space-between",
        background: "linear-gradient(to bottom, rgba(0,0,0,.6), transparent)", pointerEvents: "none", zIndex: 20
      }}>
        <div style={{ pointerEvents: "auto" }}>
          <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: ".08em", color: "#fff", opacity: .85 }}>MANTU LIGHT MESH</div>
          <div style={{ fontSize: 9, color: "rgba(255,255,255,.35)", marginTop: 1, letterSpacing: ".04em" }}>WebGL gradient generator</div>
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center", pointerEvents: "auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 4, marginRight: 6 }}>
            <span style={{ fontSize: 9, color: "rgba(255,255,255,.4)" }}>Blend</span>
            <input type="range" min="1.5" max="5" step="0.1" value={power}
              onChange={(e) => setPow(+e.target.value)} style={{ width: 60, accentColor: "#D4B8FF" }} />
          </div>
          <button style={btnS(true)} onClick={(e) => { e.stopPropagation(); shuffle(); }}>🔀 Shuffle</button>
          <button style={btnS(showFil)} onClick={(e) => { e.stopPropagation(); setShowFil(!showFil); }}>✨ Filters</button>
          <button style={btnS(false)} onClick={(e) => { e.stopPropagation(); clearAll(); }}>🗑️ Clear</button>
          <button style={btnS(false)} onClick={(e) => { e.stopPropagation(); dlPNG(); }}>⬇ PNG</button>
          <button style={btnS(false)} onClick={(e) => { e.stopPropagation(); setCss(!cssVis); }}>{cssVis ? "Hide CSS" : "{ } CSS"}</button>
        </div>
      </div>

      {/* ═══ POINT PARAMS PANEL ═══ */}
      {selData && (
        <div onClick={e => e.stopPropagation()} style={{ ...panelS, left: 16, top: 60 }}>
          <div style={panelHead}>
            Point {sel + 1}
            <span style={{ marginLeft: 8, color: selData.ck === "black" ? "#555" : PRACTICES[selData.ck]?.hex }}>
              ● {selData.ck === "black" ? "Off" : PRACTICES[selData.ck]?.label}
            </span>
          </div>
          <div style={{ display: "flex", gap: 3, marginBottom: 6 }}>
            {SHAPES.map(s => (
              <button key={s.id} onClick={e => { e.stopPropagation(); updPt(sel, "shape", s.id); }}
                title={s.n} style={{
                  width: 26, height: 24, borderRadius: 4, fontSize: s.id === 5 ? 9 : 12,
                  background: (selData.shape ?? 0) === s.id ? "rgba(212,184,255,.2)" : "rgba(255,255,255,.06)",
                  border: `1px solid ${(selData.shape ?? 0) === s.id ? "rgba(212,184,255,.4)" : "rgba(255,255,255,.08)"}`,
                  color: "#fff", cursor: "pointer", fontFamily: "inherit", padding: 0, lineHeight: 1
                }}>{s.l}</button>))}
          </div>
          {(selData.shape ?? 0) > 0 && <Sl l="Size" v={selData.shapeSize ?? .15} set={v => updPt(sel, "shapeSize", v)} mn={0.02} mx={0.5} st={.01} />}
          <Sl l="Radius" v={selData.radius ?? 0.5} set={v => updPt(sel, "radius", v)} mn={0.05} mx={1.5} st={.05} />
          <Sl l="Intensity" v={selData.intensity ?? 1} set={v => updPt(sel, "intensity", v)} mn={0.1} mx={4} st={.1} />
          <Sl l="Softness" v={selData.softness ?? 1} set={v => updPt(sel, "softness", v)} mn={0.3} mx={3} st={.1} />
          <Sl l="Stretch" v={selData.stretch ?? 1} set={v => updPt(sel, "stretch", v)} mn={0.1} mx={5} st={.1} />
          <Sl l="Angle" v={selData.angle ?? 0} set={v => updPt(sel, "angle", v)} mn={0} mx={360} st={1} />
          <div style={{ marginTop: 6, display: "flex", gap: 4 }}>
            <button onClick={e => {
              e.stopPropagation(); updPt(sel, "radius", .5); updPt(sel, "intensity", 1);
              updPt(sel, "softness", 1); updPt(sel, "stretch", 1); updPt(sel, "angle", 0);
              updPt(sel, "shape", 0); updPt(sel, "shapeSize", .15);
            }}
              style={{ ...btnS(false), padding: "3px 8px", fontSize: 9 }}>Reset</button>
            <button onClick={e => { e.stopPropagation(); remove(); }}
              style={{ ...btnS(false), padding: "3px 8px", fontSize: 9, color: "#ff6b6b", borderColor: "rgba(255,100,100,.2)" }}>
              ✕ Remove</button>
          </div>
        </div>
      )}

      {/* ═══ FILTERS PANEL ═══ */}
      {showFil && (
        <div onClick={e => e.stopPropagation()} style={{ ...panelS, left: 16, top: selData ? 320 : 60 }}>
          <div style={panelHead}>Global Filters</div>
          <Sl l="Contrast" v={fil.contrast} set={v => updFil("contrast", v)} mn={0.5} mx={2.5} st={.05} />
          <Sl l="Bloom" v={fil.bloom} set={v => updFil("bloom", v)} mn={0} mx={2} st={.05} />
          <Sl l="Grain" v={fil.grain} set={v => updFil("grain", v)} mn={0} mx={0.3} st={.01} />
          <Sl l="Chromatic" v={fil.chroma} set={v => updFil("chroma", v)} mn={0} mx={0.08} st={.002} />
          <Sl l="Vignette" v={fil.vignette} set={v => updFil("vignette", v)} mn={0} mx={3} st={.1} />
          <div style={{ borderTop: "1px solid rgba(255,255,255,.06)", margin: "6px 0", paddingTop: 6 }}>
            <div style={{ fontSize: 8, color: "rgba(255,255,255,.3)", marginBottom: 4 }}>SCAN LINES</div>
            <Sl l="Frequency" v={fil.scanFreq} set={v => updFil("scanFreq", v)} mn={5} mx={200} st={1} />
            <Sl l="Intensity" v={fil.scanInt} set={v => updFil("scanInt", v)} mn={0} mx={1} st={.05} />
          </div>
          <button onClick={e => { e.stopPropagation(); setFil({ contrast: 1, bloom: 0, grain: 0, chroma: 0, vignette: 1, scanFreq: 50, scanInt: 0 }); }}
            style={{ ...btnS(false), padding: "3px 8px", fontSize: 9, marginTop: 4 }}>Reset Filters</button>
        </div>
      )}

      {/* ═══ BOTTOM PALETTE ═══ */}
      <div style={{
        position: "absolute", bottom: 0, left: 0, right: 0, padding: "12px 0 16px",
        display: "flex", flexDirection: "column", alignItems: "center", gap: 8,
        background: "linear-gradient(to top, rgba(0,0,0,.7), transparent)", pointerEvents: "none", zIndex: 20
      }}>
        <div style={{ display: "flex", gap: 12, pointerEvents: "auto" }}>
          <div onClick={e => { e.stopPropagation(); assign("black"); }}
            style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4, cursor: "pointer" }}>
            <div style={{
              width: 36, height: 36, borderRadius: 8, background: "#222",
              border: `2px solid ${selData?.ck === "black" ? "#fff" : "rgba(255,255,255,.15)"}`, transition: "all .15s"
            }} />
            <span style={{ fontSize: 8, color: "rgba(255,255,255,.35)", letterSpacing: ".04em" }}>Off</span>
          </div>
          {P_KEYS.map(k => {
            const p = PRACTICES[k]; const isA = active.includes(k); const isAs = selData?.ck === k;
            return (<div key={k} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <div onClick={e => { e.stopPropagation(); assign(k); }}
                style={{
                  width: 36, height: 36, borderRadius: 8, background: p.hex, cursor: "pointer",
                  border: `2px solid ${isAs ? "#fff" : "rgba(255,255,255,.15)"}`,
                  opacity: isA ? 1 : .35, transition: "all .15s", boxShadow: isAs ? `0 0 12px ${p.hex}66` : "none"
                }} />
              <span style={{
                fontSize: 8, color: isA ? "rgba(255,255,255,.55)" : "rgba(255,255,255,.25)",
                letterSpacing: ".04em"
              }}>{p.label}</span>
              <button onClick={e => { e.stopPropagation(); toggleAct(k); }}
                style={{
                  fontSize: 9, padding: "1px 6px", cursor: "pointer", fontFamily: "inherit",
                  background: isA ? "rgba(255,255,255,.1)" : "transparent",
                  border: `1px solid ${isA ? p.hex + "66" : "rgba(255,255,255,.08)"}`, borderRadius: 4,
                  color: isA ? p.hex : "rgba(255,255,255,.25)", transition: "all .15s"
                }}>
                {isA ? "🔒" : "🔓"}</button>
            </div>);
          })}
        </div>
        <div style={{ fontSize: 8, color: "rgba(255,255,255,.2)", letterSpacing: ".05em" }}>
          Click canvas to add · Drag to move · Double-click to cycle color · {active.length} color{active.length > 1 ? "s" : ""} locked for shuffle
        </div>
      </div>

      {/* ═══ CSS PANEL ═══ */}
      {cssVis && (
        <div onClick={e => e.stopPropagation()} style={{
          position: "absolute", right: 16, top: 60, bottom: 120, width: 320,
          background: "rgba(10,10,10,.92)", border: "1px solid rgba(255,255,255,.08)", borderRadius: 10, zIndex: 30,
          display: "flex", flexDirection: "column", backdropFilter: "blur(16px)", WebkitBackdropFilter: "blur(16px)"
        }}>
          <div style={{
            padding: "10px 14px", borderBottom: "1px solid rgba(255,255,255,.06)",
            display: "flex", justifyContent: "space-between", alignItems: "center"
          }}>
            <span style={{ fontSize: 10, color: "rgba(255,255,255,.5)", letterSpacing: ".06em" }}>CSS EXPORT</span>
            <button onClick={e => { e.stopPropagation(); cpCSS(); }} style={btnS(false)}>Copy</button>
          </div>
          <pre style={{
            flex: 1, overflow: "auto", padding: 14, margin: 0, fontSize: 10, color: "rgba(255,255,255,.55)",
            lineHeight: 1.5, fontFamily: "'IBM Plex Mono', monospace", whiteSpace: "pre-wrap"
          }}>{css}</pre>
        </div>)}
      {/* ═══ TUTORIAL OVERLAY ═══ */}
      {
        tutStep >= 0 && (
          <div style={{
            position: "absolute", inset: 0, zIndex: 100, pointerEvents: "auto",
            background: tutStep === 0 ? "rgba(0,0,0,.4)" : "transparent"
          }}>
            {/* Step 0: Welcome (Center) */}
            {tutStep === 0 && (
              <div style={{
                position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)",
                width: 300, padding: 20, background: "#111", border: "1px solid #333", borderRadius: 12,
                boxShadow: "0 20px 50px rgba(0,0,0,.5)", color: "#eee"
              }}>
                <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 8, color: "#D4B8FF" }}>Welcome to Mantu Light Mesh</div>
                <p style={{ fontSize: 11, opacity: .7, lineHeight: 1.5, marginBottom: 16 }}>
                  Create beautiful, eclipse-style gradients with solid shapes. Here is a quick guide to get you started.
                </p>
                <div style={{ display: "flex", justifyContent: "flex-end" }}>
                  <button onClick={nextTut} style={{ ...btnS(true), background: "#D4B8FF", color: "#000", border: "none" }}>Start Tour</button>
                </div>
              </div>
            )}

            {/* Step 1: Canvas Interaction (Near Center/Point) */}
            {tutStep === 1 && (
              <div style={{
                position: "absolute", top: "calc(50% + 40px)", left: "50%", transform: "translateX(-50%)",
                width: 260, padding: 14, background: "#111", border: "1px solid #333", borderRadius: 8,
                boxShadow: "0 10px 30px rgba(0,0,0,.5)", color: "#eee"
              }}>
                <div style={{ width: 0, height: 0, border: "6px solid transparent", borderBottomColor: "#333", position: "absolute", top: -12, left: "50%", transform: "translateX(-50%)" }} />
                <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 4 }}>Interact with Light</div>
                <p style={{ fontSize: 10, opacity: .7, lineHeight: 1.4, marginBottom: 12 }}>
                  • <b>Click</b> anywhere to add a light.<br />
                  • <b>Drag</b> points to move them.<br />
                  • <b>Double-click</b> a point to cycle colors.<br />
                  • <b>Click</b> a point to edit properties.
                </p>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <button onClick={skipTut} style={{ fontSize: 10, color: "#666", background: "transparent", border: "none", cursor: "pointer" }}>Skip</button>
                  <button onClick={nextTut} style={{ ...btnS(true), padding: "4px 10px", fontSize: 10 }}>Next</button>
                </div>
              </div>
            )}

            {/* Step 2: Top Toolbar (Top Right) */}
            {tutStep === 2 && (
              <div style={{
                position: "absolute", top: 60, right: 20,
                width: 240, padding: 14, background: "#111", border: "1px solid #333", borderRadius: 8,
                boxShadow: "0 10px 30px rgba(0,0,0,.5)", color: "#eee"
              }}>
                <div style={{ width: 0, height: 0, border: "6px solid transparent", borderBottomColor: "#333", position: "absolute", top: -12, right: 40 }} />
                <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 4 }}>Refine & Export</div>
                <p style={{ fontSize: 10, opacity: .7, lineHeight: 1.4, marginBottom: 12 }}>
                  Adjust global <b>Blend</b> power, apply <b>Effects</b> (grain, bloom), and <b>Export</b> your creation as CSS or PNG.
                </p>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <button onClick={skipTut} style={{ fontSize: 10, color: "#666", background: "transparent", border: "none", cursor: "pointer" }}>Skip</button>
                  <button onClick={nextTut} style={{ ...btnS(true), padding: "4px 10px", fontSize: 10 }}>Next</button>
                </div>
              </div>
            )}

            {/* Step 3: Bottom Palette (Bottom Center) */}
            {tutStep === 3 && (
              <div style={{
                position: "absolute", bottom: 90, left: "50%", transform: "translateX(-50%)",
                width: 260, padding: 14, background: "#111", border: "1px solid #333", borderRadius: 8,
                boxShadow: "0 10px 30px rgba(0,0,0,.5)", color: "#eee"
              }}>
                <div style={{ width: 0, height: 0, border: "6px solid transparent", borderTopColor: "#333", position: "absolute", bottom: -12, left: "50%", transform: "translateX(-50%)" }} />
                <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 4 }}>Mantu Palette</div>
                <p style={{ fontSize: 10, opacity: .7, lineHeight: 1.4, marginBottom: 12 }}>
                  Use the official Mantu brand colors. Click the 🔓 icon to lock a color before shuffling to keep it in your design.
                </p>
                <div style={{ display: "flex", justifyContent: "flex-end" }}>
                  <button onClick={nextTut} style={{ ...btnS(true), padding: "4px 10px", fontSize: 10, background: "#D4B8FF", color: "#000", border: "none" }}>Finish</button>
                </div>
              </div>
            )}
          </div>
        )
      }
    </div >
  );
}
