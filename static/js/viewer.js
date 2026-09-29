/* PACT interactive 3D viewers (three.js, vendored; see the import map in index.html).
 *
 * Each [data-viewer] card loads bundles from static/viewer/<set>/<stem>/:
 *   meta.json  layout of data.bin and clip metadata (see the bundle spec)
 *   data.bin   little-endian typed arrays at the byte offsets listed in meta.arrays
 *   video.mp4  the input video; its currentTime drives the 3D frame
 * The body is a SkinnedMesh with flat bones (every bone a direct child of one root at the identity), bound with
 * bone inverses = inverse(translation(rest_joints[k])), so a bone's world pose per row is exactly
 * (joint_quat[n,k], joint_pos[n,k]) and three.js performs linear blend skinning.
 */
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

const BODY_COLOR = "rgb(166, 189, 219)";
const ARROW = { shaftR: 0.032, headR: 0.075, headLen: 0.17, minBW: 0.05 };
const CONTACT_R = 0.012;
const POINT_SIZE = 0.02;
const FRUSTUM_DEPTH = 0.3;
const FOV = 60;
// Default framing: the orbit target sits TARGET_LIFT above the pelvis and follows it; the view fits
// FIT_HALF metres around the target (body plus a 1 m force arrow, about 3 m away on a 16:10 stage),
// seen from ELEVATION_DEG above, looking slightly down.
const FIT_HALF = 1.7;
const TARGET_LIFT = 0.4;
const ELEVATION_DEG = 8;
const FOLLOW_TAU = 0.12;   // smoothing time constant of the follow target (s)
const UP = new THREE.Vector3(0, 1, 0);
const DTYPES = { float32: Float32Array, uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array };

// Point colours arrive as sRGB bytes; three.js expects linear vertex colours.
const SRGB_TO_LINEAR = new Float32Array(256).map((_, i) => {
  const c = i / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
});

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const coarsePointer = window.matchMedia("(pointer: coarse)").matches;

function readArrays(meta, buf) {
  const out = {};
  for (const [name, a] of Object.entries(meta.arrays)) {
    const Ctor = DTYPES[a.dtype];
    if (!Ctor) throw new Error(`unsupported dtype ${a.dtype} for ${name}`);
    if (a.offset % 4 !== 0 || a.offset + a.nbytes > buf.byteLength) throw new Error(`bad layout for ${name}`);
    out[name] = new Ctor(buf, a.offset, a.nbytes / Ctor.BYTES_PER_ELEMENT);
  }
  return out;
}

async function fetchOk(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res;
}

/* Frustum outline in the OpenCV camera frame (looks along +Z, image +Y down), with a small
 * triangle on the image's top edge. Placing it with cam_quat (camera-to-world, OpenCV) needs no flip. */
function frustumGeometry(fovYDeg, aspect, depth) {
  const h = depth * Math.tan(THREE.MathUtils.degToRad(fovYDeg) / 2);
  const w = h * aspect;
  const c = [[-w, -h, depth], [w, -h, depth], [w, h, depth], [-w, h, depth]];
  const p = [];
  for (let i = 0; i < 4; i++) p.push(0, 0, 0, ...c[i], ...c[i], ...c[(i + 1) % 4]);
  const t = [[-0.45 * w, -h, depth], [0, -h - 0.5 * h, depth], [0.45 * w, -h, depth]];
  p.push(...t[0], ...t[1], ...t[1], ...t[2]);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
  return g;
}

class Viewer {
  constructor(root) {
    this.root = root;
    this.set = root.dataset.set;
    this.forceColor = new THREE.Color(root.dataset.forceColor);
    root.style.setProperty("--accent", root.dataset.forceColor);

    this.stage = root.querySelector(".viewer__stage");
    this.status = root.querySelector(".viewer__status");
    this.video = root.querySelector(".viewer__video video");
    this.scrub = root.querySelector(".viewer__scrub");
    this.playBtn = root.querySelector('[data-action="play"]');
    this.chips = [...root.querySelectorAll(".vchip")];
    this.toggles = [...root.querySelectorAll(".vtoggle")];
    this.layerOn = {};
    for (const t of this.toggles) this.layerOn[t.dataset.layer] = t.getAttribute("aria-pressed") === "true";

    this.clip = null;
    this.loadToken = 0;
    this.started = false;
    this.visible = false;
    this.userPaused = reduceMotion;
    this.row = -1;
    this.raf = 0;
    this.needsRender = true;
    this.scrubbing = false;
    // Clock used only if the video cannot be played (the 3D still animates).
    this.videoOk = true;
    this.clockPlaying = false;
    this.clockTime = 0;
    this.lastNow = 0;

    this.updatePlayButton();
  }

  /* ----- Setup (runs once, when the card first comes near the viewport) ----- */
  init() {
    if (this.started) return;
    this.started = true;

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    } catch (e) {
      this.setStatus("3D view unavailable in this browser");
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0xffffff, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer = renderer;
    const canvas = renderer.domElement;
    canvas.setAttribute("aria-label", "3D scene");
    this.stage.prepend(canvas);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xffffff);
    this.scene.fog = new THREE.Fog(0xffffff, 8, 24); // fades the far floor and scene; follows the orbit distance
    this.camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 200);
    this.scene.add(this.camera);

    // Key / fill / rim lights ride with the camera so every orbit angle stays lit; soft sky light fills the rest.
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xdfe3ea, 1.15));
    const aim = new THREE.Object3D();
    aim.position.set(0, 0, -4);
    this.camera.add(aim);
    for (const [intensity, x, y, z] of [[2.2, -2.5, 3, 1], [0.7, 3, 0.5, 0.5], [1.4, 0.5, 2.5, -8]]) {
      const light = new THREE.DirectionalLight(0xffffff, intensity);
      light.position.set(x, y, z);
      light.target = aim;
      this.camera.add(light);
    }

    const controls = new OrbitControls(this.camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;
    controls.enableZoom = false; // enabled once the scene is clicked, so the page still scrolls past it
    controls.addEventListener("change", () => { this.needsRender = true; });
    this.controls = controls;

    if (coarsePointer) {
      // Touch: the canvas lets the page scroll until it is tapped.
      this.root.classList.add("is-touch");
      controls.enabled = false;
      canvas.style.touchAction = "pan-y";
      canvas.addEventListener("click", () => this.engage());
      document.addEventListener("pointerdown", (e) => {
        if (!this.stage.contains(e.target)) this.disengage();
      }, true);
    } else {
      canvas.addEventListener("pointerdown", () => this.engage());
      canvas.addEventListener("pointerleave", () => this.disengage());
    }

    new ResizeObserver(() => this.resize()).observe(this.stage);
    this.resize();
    this.root.classList.add("is-ready");

    // Video drives the 3D frame.
    this.video.loop = true;
    this.video.muted = true;
    this.video.addEventListener("play", () => this.updatePlayButton());
    this.video.addEventListener("pause", () => this.updatePlayButton());
    this.video.addEventListener("error", () => {
      if (!this.video.getAttribute("src")) return;
      this.videoOk = false; // no video pane: a clock drives the 3D instead
      this.clockPlaying = this.visible && !this.userPaused;
      this.updatePlayButton();
    });

    this.playBtn.addEventListener("click", () => {
      if (this.isPlaying()) {
        this.userPaused = true;
        this.pauseMedia();
      } else {
        this.userPaused = false;
        this.playMedia();
      }
    });
    this.root.querySelector('[data-action="reset"]').addEventListener("click", () => this.resetView());

    const endScrub = () => {
      if (!this.scrubbing) return;
      this.scrubbing = false;
      if (this.wasPlaying) this.playMedia();
    };
    this.scrub.addEventListener("pointerdown", () => {
      this.scrubbing = true;
      this.wasPlaying = this.isPlaying();
      this.pauseMedia();
    });
    this.scrub.addEventListener("input", () => this.seekRow(Number(this.scrub.value)));
    this.scrub.addEventListener("change", endScrub);
    this.scrub.addEventListener("pointerup", endScrub);

    for (const chip of this.chips) {
      chip.addEventListener("click", () => {
        if (chip.getAttribute("aria-pressed") !== "true") this.loadClip(chip.dataset.clip);
      });
    }
    for (const t of this.toggles) {
      t.addEventListener("click", () => {
        const on = t.getAttribute("aria-pressed") !== "true";
        t.setAttribute("aria-pressed", String(on));
        this.layerOn[t.dataset.layer] = on;
        this.applyLayers();
      });
    }

    const first = this.chips.find((c) => c.getAttribute("aria-pressed") === "true") || this.chips[0];
    this.loadClip(first.dataset.clip);
    if (this.visible) this.start();
  }

  engage() {
    this.controls.enableZoom = true;
    if (coarsePointer && !this.controls.enabled) {
      this.controls.enabled = true;
      this.renderer.domElement.style.touchAction = "none";
    }
    this.root.classList.add("is-engaged");
  }

  disengage() {
    if (!this.controls) return;
    this.controls.enableZoom = false;
    if (coarsePointer) {
      this.controls.enabled = false;
      this.renderer.domElement.style.touchAction = "pan-y";
    }
    this.root.classList.remove("is-engaged");
  }

  resize() {
    const w = this.stage.clientWidth;
    const h = this.stage.clientHeight;
    if (!w || !h) return;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.needsRender = true;
  }

  setStatus(text) {
    this.status.hidden = !text;
    if (text) this.status.textContent = text;
  }

  /* ----- Clip loading ----- */
  async loadClip(stem) {
    const token = ++this.loadToken;
    for (const c of this.chips) c.setAttribute("aria-pressed", String(c.dataset.clip === stem));
    this.setStatus("Loading…");

    const base = `static/viewer/${this.set}/${stem}/`;
    this.video.pause();
    this.videoOk = true;
    this.clockPlaying = false;
    this.clockTime = 0;
    this.video.poster = base + "video.jpg";
    this.video.src = base + "video.mp4";
    this.video.preload = "auto";

    try {
      const meta = await (await fetchOk(base + "meta.json")).json();
      const buf = await (await fetchOk(base + "data.bin")).arrayBuffer();
      if (token !== this.loadToken) return;
      const clip = this.buildClip(meta, buf);
      this.disposeClip();
      this.clip = clip;
      this.scene.add(clip.group);

      this.scrub.max = String(clip.N - 1);
      this.scrub.value = "0";
      for (const t of this.toggles) t.hidden = !clip.layers[t.dataset.layer];
      this.applyLayers();
      this.row = -1;
      this.resetView();
      this.setStatus(null);
      if (this.visible && !this.userPaused) this.playMedia();
    } catch (err) {
      if (token !== this.loadToken) return;
      console.warn(`[viewer] ${this.set}/${stem} unavailable:`, err.message || err);
      this.disposeClip();
      this.setStatus("3D data unavailable");
    }
  }

  buildClip(meta, buf) {
    const A = readArrays(meta, buf);
    const N = meta.n_frames;
    const J = meta.n_joints;
    const L = meta.n_limbs;
    const K = meta.n_contact_frames;
    const group = new THREE.Group();
    const layers = {};

    // Body: skinned mesh over flat bones.
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(A.verts, 3));
    geo.setIndex(new THREE.BufferAttribute(A.faces, 1));
    geo.setAttribute("skinIndex", new THREE.BufferAttribute(A.skin_idx, 4));
    geo.setAttribute("skinWeight", new THREE.BufferAttribute(A.skin_w, 4));
    geo.computeVertexNormals();

    const rig = new THREE.Object3D();
    const bones = [];
    const inverses = [];
    for (let k = 0; k < J; k++) {
      const bone = new THREE.Bone();
      bone.name = (meta.joint_names && meta.joint_names[k]) || `joint_${k}`;
      rig.add(bone);
      bones.push(bone);
      inverses.push(new THREE.Matrix4().makeTranslation(-A.rest_joints[3 * k], -A.rest_joints[3 * k + 1], -A.rest_joints[3 * k + 2]));
    }
    const skeleton = new THREE.Skeleton(bones, inverses);
    const mesh = new THREE.SkinnedMesh(geo, new THREE.MeshStandardMaterial({
      color: BODY_COLOR, roughness: 0.6, metalness: 0, flatShading: false,
    }));
    mesh.bind(skeleton, new THREE.Matrix4()); // identity bind matrix; keeps the given bone inverses
    mesh.frustumCulled = false; // the rest-pose bounds do not follow the motion
    group.add(rig, mesh);

    // Force arrows: one per limb, unit shaft along +Y scaled per frame.
    const arrowMat = new THREE.MeshStandardMaterial({ color: this.forceColor, roughness: 0.45, metalness: 0 });
    const shaftGeo = new THREE.CylinderGeometry(ARROW.shaftR, ARROW.shaftR, 1, 18, 1).translate(0, 0.5, 0);
    const headGeo = new THREE.ConeGeometry(ARROW.headR, ARROW.headLen, 28, 1).translate(0, ARROW.headLen / 2, 0);
    const forces = new THREE.Group();
    const arrows = [];
    for (let l = 0; l < L; l++) {
      const a = new THREE.Group();
      const shaft = new THREE.Mesh(shaftGeo, arrowMat);
      const head = new THREE.Mesh(headGeo, arrowMat);
      a.add(shaft, head);
      a.visible = false;
      forces.add(a);
      arrows.push({ a, shaft, head });
    }
    group.add(forces);
    layers.forces = forces;

    // Contact points: instanced spheres, only those in contact are drawn.
    const contacts = new THREE.InstancedMesh(
      new THREE.SphereGeometry(CONTACT_R, 12, 8),
      new THREE.MeshStandardMaterial({ color: this.forceColor, roughness: 0.5, metalness: 0 }),
      K,
    );
    contacts.count = 0;
    contacts.frustumCulled = false;
    group.add(contacts);
    layers.contacts = contacts;

    // Annotation set: coloured scene point cloud.
    if (meta.has_scene && A.scene_points && A.scene_colors) {
      const pg = new THREE.BufferGeometry();
      pg.setAttribute("position", new THREE.BufferAttribute(A.scene_points, 3));
      const col = new Float32Array(A.scene_colors.length);
      for (let i = 0; i < col.length; i++) col[i] = SRGB_TO_LINEAR[A.scene_colors[i]];
      pg.setAttribute("color", new THREE.BufferAttribute(col, 3));
      const points = new THREE.Points(pg, new THREE.PointsMaterial({
        size: POINT_SIZE, sizeAttenuation: true, vertexColors: true,
      }));
      group.add(points);
      layers.scene = points;
    }

    // Prediction set: a soft floor grid with half-metre cells.
    if (meta.floor_y !== null && meta.floor_y !== undefined) {
      const size = Math.max(16, 2 * Math.ceil(meta.radius * 3));
      const grid = new THREE.GridHelper(size, size * 2, 0xd9d9d9, 0xe8e8e8);
      grid.material.transparent = true;
      grid.material.opacity = 0.9;
      grid.material.depthWrite = false;
      grid.position.set(Math.round(meta.center[0] * 2) / 2, meta.floor_y, Math.round(meta.center[2] * 2) / 2);
      group.add(grid);
      layers.floor = grid;
    }

    // Camera path (moving cameras) and the current camera frustum (all clips with a camera pose).
    let frustum = null;
    if (A.cam_pos && A.cam_quat) {
      const cam = new THREE.Group();
      if (meta.has_camera_path) {
        const pathGeo = new THREE.BufferGeometry();
        pathGeo.setAttribute("position", new THREE.BufferAttribute(A.cam_pos, 3));
        cam.add(new THREE.Line(pathGeo, new THREE.LineBasicMaterial({ color: 0xa0a4ab })));
      }
      const fov = (meta.camera && meta.camera.fov_y_deg) || 55;
      const aspect = (meta.camera && meta.camera.aspect) || (meta.video.width / meta.video.height);
      frustum = new THREE.LineSegments(frustumGeometry(fov, aspect, FRUSTUM_DEPTH),
        new THREE.LineBasicMaterial({ color: 0x3a3a3a }));
      cam.add(frustum);
      group.add(cam);
      layers.camera = cam;
    }

    // Video frame -> row lookup (rows may start later or skip frames; frames before the first row use row 0).
    const pelvis = Math.max(0, (meta.joint_names || []).indexOf("pelvis"));
    const fi = meta.frame_indices || Array.from({ length: N }, (_, i) => i);
    const nVid = Math.max((meta.video && meta.video.n_frames) || 0, fi[N - 1] + 1);
    const frameToRow = new Int32Array(nVid);
    for (let f = 0, r = 0; f < nVid; f++) {
      while (r + 1 < N && fi[r + 1] <= f) r++;
      frameToRow[f] = r;
    }

    return {
      meta, A, N, J, L, K, group, layers, mesh, bones, skeleton, arrows, contacts, frustum, pelvis,
      fi, frameToRow, fps: (meta.video && meta.video.fps) || meta.fps,
    };
  }

  disposeClip() {
    const c = this.clip;
    if (!c) return;
    this.scene.remove(c.group);
    c.group.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) [].concat(o.material).forEach((m) => m.dispose());
      if (o.isInstancedMesh) o.dispose();
    });
    c.skeleton.dispose();
    this.clip = null;
    this.needsRender = true;
  }

  applyLayers() {
    const c = this.clip;
    if (!c) return;
    for (const [name, obj] of Object.entries(c.layers)) obj.visible = this.layerOn[name] !== false;
    this.needsRender = true;
  }

  /* ----- Per-row pose ----- */
  setRow(n) {
    const c = this.clip;
    const A = c.A;
    const valid = A.valid ? A.valid[n] !== 0 : true;
    c.mesh.visible = valid;
    if (valid) {
      for (let k = 0; k < c.J; k++) {
        const i = n * c.J + k;
        c.bones[k].position.fromArray(A.joint_pos, 3 * i);
        c.bones[k].quaternion.fromArray(A.joint_quat, 4 * i);
      }
    }

    const scale = c.meta.arrow_m_per_bw || 0.7;
    const dir = new THREE.Vector3();
    for (let l = 0; l < c.L; l++) {
      const { a, shaft, head } = c.arrows[l];
      const i = 3 * (n * c.L + l);
      dir.fromArray(A.limb_force, i);
      const mag = dir.length();
      if (!valid || !(mag >= ARROW.minBW)) {
        a.visible = false;
        continue;
      }
      const len = mag * scale;
      const s = Math.min(1, len / (1.6 * ARROW.headLen)); // shrink the head on short arrows
      const headLen = ARROW.headLen * s;
      a.visible = true;
      a.position.fromArray(A.limb_anchor, i);
      a.quaternion.setFromUnitVectors(UP, dir.divideScalar(mag));
      shaft.scale.set(s, Math.max(len - headLen, 1e-4), s);
      head.scale.setScalar(s);
      head.position.y = len - headLen;
    }

    const m = new THREE.Matrix4();
    let count = 0;
    if (valid && A.frame_contact && A.frame_pts) {
      for (let k = 0; k < c.K; k++) {
        const i = n * c.K + k;
        if (!A.frame_contact[i]) continue;
        m.makeTranslation(A.frame_pts[3 * i], A.frame_pts[3 * i + 1], A.frame_pts[3 * i + 2]);
        c.contacts.setMatrixAt(count++, m);
      }
    }
    c.contacts.count = count;
    c.contacts.instanceMatrix.needsUpdate = true;

    if (c.frustum) {
      c.frustum.position.fromArray(A.cam_pos, 3 * n);
      c.frustum.quaternion.fromArray(A.cam_quat, 4 * n);
    }
  }

  /* ----- Time: the video's currentTime selects the row ----- */
  currentTime() {
    return this.videoOk ? this.video.currentTime : this.clockTime;
  }

  currentRow() {
    const c = this.clip;
    const f = Math.floor(this.currentTime() * c.fps + 1e-3);
    return c.frameToRow[Math.min(Math.max(f, 0), c.frameToRow.length - 1)];
  }

  seekRow(r) {
    const c = this.clip;
    if (!c) return;
    const t = (c.fi[r] + 0.25) / c.fps; // inside source frame fi[r]
    if (this.videoOk) this.video.currentTime = t;
    else this.clockTime = t;
    this.showRow(r);
  }

  showRow(r) {
    if (r === this.row) return;
    this.row = r;
    this.setRow(r);
    this.needsRender = true;
    if (!this.scrubbing) this.scrub.value = String(r);
    this.scrub.style.setProperty("--p", `${(100 * r) / Math.max(1, this.clip.N - 1)}%`);
  }

  isPlaying() {
    return this.videoOk ? !this.video.paused : this.clockPlaying;
  }

  playMedia() {
    if (this.videoOk) {
      const p = this.video.play();
      if (p && p.catch) p.catch(() => {});
    } else {
      this.clockPlaying = true;
    }
    this.updatePlayButton();
  }

  pauseMedia() {
    this.video.pause();
    this.clockPlaying = false;
    this.updatePlayButton();
  }

  updatePlayButton() {
    const playing = this.isPlaying();
    this.playBtn.classList.toggle("is-paused", !playing);
    this.playBtn.setAttribute("aria-label", playing ? "Pause" : "Play");
  }

  /* ----- Camera framing: orbit around the (smoothed) pelvis ----- */
  pelvisAt(row, out) {
    const c = this.clip;
    if (c.A.valid && !c.A.valid[row]) return null;
    return out.fromArray(c.A.joint_pos, 3 * (row * c.J + c.pelvis));
  }

  // Move target and camera together, so the user's orbit angle, distance and pan offset are kept.
  follow(dt) {
    const p = this.pelvisAt(Math.max(this.row, 0), new THREE.Vector3());
    if (!p) return;
    const k = dt === null ? 1 : 1 - Math.exp(-dt / FOLLOW_TAU);
    const delta = p.sub(this.followPos).multiplyScalar(k);
    if (delta.lengthSq() < 1e-12) return;
    this.followPos.add(delta);
    this.controls.target.add(delta);
    this.camera.position.add(delta);
    this.needsRender = true;
  }

  resetView() {
    const c = this.clip;
    if (!c) return;
    const row = this.row >= 0 ? this.row : this.currentRow();
    const pelvis = this.pelvisAt(row, new THREE.Vector3()) || new THREE.Vector3().fromArray(c.meta.center);
    this.followPos = pelvis.clone();
    const center = pelvis.clone().addScaledVector(UP, TARGET_LIFT);
    // Start roughly from the recording camera's side, turned a little for a three-quarter view.
    let dir = null;
    if (c.A.cam_pos) {
      dir = new THREE.Vector3().fromArray(c.A.cam_pos, 0).sub(center);
      dir.y = 0;
      dir = dir.lengthSq() > 1e-6 ? dir.normalize() : null;
    }
    dir = dir || new THREE.Vector3(0, 0, 1);
    dir.applyAxisAngle(UP, THREE.MathUtils.degToRad(24));
    const elev = THREE.MathUtils.degToRad(ELEVATION_DEG);
    dir.multiplyScalar(Math.cos(elev)).addScaledVector(UP, Math.sin(elev));

    const vFov = THREE.MathUtils.degToRad(this.camera.fov);
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * this.camera.aspect);
    const dist = FIT_HALF / Math.tan(Math.min(vFov, hFov) / 2);
    this.camera.position.copy(center).addScaledVector(dir, dist);
    this.controls.target.copy(center);
    this.controls.minDistance = 0.3;
    this.controls.maxDistance = dist * 4;
    this.controls.update();
    this.needsRender = true;
  }

  /* ----- Render loop: runs only while the card is on screen ----- */
  // A clip without a video pane plays on a clock; check again whenever the viewer comes back into view.
  async retryVideo() {
    const src = this.video.getAttribute("src");
    if (this.videoOk || !src || this.retrying) return;
    this.retrying = true;
    try {
      const res = await fetch(src, { method: "HEAD", cache: "no-store" });
      if (res.ok && !this.videoOk && this.video.getAttribute("src") === src) {
        const t = this.clockTime;
        this.videoOk = true;
        this.video.load();
        this.video.currentTime = t;
        if (this.clockPlaying) this.playMedia();
        this.clockPlaying = false;
      }
    } catch (e) { /* stay on the clock */ }
    this.retrying = false;
  }

  start() {
    this.retryVideo();
    if (!this.renderer || this.raf) return;
    this.lastNow = performance.now();
    this.raf = requestAnimationFrame(this.tick);
    if (!this.userPaused) this.playMedia();
  }

  stop() {
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
    if (this.renderer) {
      this.pauseMedia();
      this.disengage();
    }
  }

  tick = (now) => {
    this.raf = requestAnimationFrame(this.tick);
    const dt = Math.min(0.1, (now - this.lastNow) / 1000);
    this.lastNow = now;
    const c = this.clip;
    if (c) {
      if (!this.videoOk && this.clockPlaying) {
        this.clockTime = (this.clockTime + dt) % (c.frameToRow.length / c.fps);
      }
      this.showRow(this.currentRow());
      this.follow(dt);
    }
    if (this.controls.update()) this.needsRender = true;
    if (this.needsRender) {
      const d = this.camera.position.distanceTo(this.controls.target);
      this.scene.fog.near = 1.6 * d;
      this.scene.fog.far = 4.5 * d;
      this.renderer.render(this.scene, this.camera);
      this.needsRender = false;
    }
  };
}

/* ----- Boot: lazy WebGL init near the viewport, run only while visible ----- */
const viewers = [...document.querySelectorAll("[data-viewer]")].map((el) => new Viewer(el));
window.__pactViewers = viewers; // handle for inspection from the browser console

if (viewers.length) {
  const near = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      near.unobserve(e.target);
      viewers.find((v) => v.root === e.target).init();
    }
  }, { rootMargin: "400px 0px" });

  const onScreen = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const v = viewers.find((x) => x.root === e.target);
      v.visible = e.isIntersecting;
      if (v.visible) v.start();
      else v.stop();
    }
  }, { threshold: 0 });

  for (const v of viewers) {
    near.observe(v.root);
    onScreen.observe(v.root);
  }
}
