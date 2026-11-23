// Visualiseur 3D de la main robotique (Three.js).
// Reçoit via WebSocket : { flex: [5 × 0–180], percent: [...], servo: [...], hand: bool }
// Ordre des doigts : pouce, index, majeur, annulaire, auriculaire.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";

const DEG = Math.PI / 180;

// ---------------------------------------------------------------- Scène
const container = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1d2127);

const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(12, 16, 38);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 8, 0);
controls.enableDamping = true;
controls.update();

scene.add(new THREE.HemisphereLight(0xfff6e8, 0x2a2f36, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(8, 20, 14);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
Object.assign(sun.shadow.camera, { left: -15, right: 15, top: 25, bottom: -10 });
scene.add(sun);

const ground = new THREE.Mesh(
  new THREE.CircleGeometry(16, 64),
  new THREE.MeshStandardMaterial({ color: 0x2a2f36, roughness: 1 })
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -6;
ground.receiveShadow = true;
scene.add(ground);

// ---------------------------------------------------------- Matériaux
// Beige mat façon PLA imprimé en 3D, axes d'articulation un peu plus sombres.
const plastic = new THREE.MeshStandardMaterial({ color: 0xe6d6b0, roughness: 0.85, metalness: 0 });
const pinMat = new THREE.MeshStandardMaterial({ color: 0x8c8172, roughness: 0.6, metalness: 0.3 });

function shadowed(mesh) {
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

// ------------------------------------------------------- Construction
// Repère : les doigts pointent vers +Y, la paume regarde vers +Z.
// Plier un doigt = rotation positive autour de X (le bout part vers +Z).
// Le modèle est construit en main gauche (pouce en +X) puis retourné en
// miroir : on obtient une main DROITE, comme votre main droite vue en miroir
// dans la fenêtre webcam. Les rotations restent exprimées « vers le pouce ».
const hand = new THREE.Group();
hand.scale.x = -1;
scene.add(hand);

// Avant-bras (fixe).
const forearm = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(2.6, 2.9, 5, 32), plastic));
forearm.position.y = -2.5;
forearm.scale.z = 0.65;
hand.add(forearm);

// Poignet : rotation autour de l'avant-bras (axe Y), puis inclinaison (axe Z,
// articulation au sommet de l'avant-bras). La paume est accrochée à l'inclinaison.
const wristRoll = new THREE.Group();
hand.add(wristRoll);
const wristDev = new THREE.Group();
wristRoll.add(wristDev);
const wristPin = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(0.7, 0.7, 3.2, 16), pinMat));
wristPin.rotation.x = Math.PI / 2;  // axe d'inclinaison, perpendiculaire à la paume
wristDev.add(wristPin);

// Paume.
const palmW = 8.4, palmH = 8.8, palmD = 2.6;
const palm = shadowed(new THREE.Mesh(new RoundedBoxGeometry(palmW, palmH, palmD, 4, 0.7), plastic));
palm.position.y = palmH / 2 + 0.4;
wristDev.add(palm);

/** Pose le poignet : rotation et inclinaison en degrés, positives vers le pouce. */
function setWrist(rotationDeg, deviationDeg) {
  wristRoll.rotation.y = rotationDeg * DEG;    // la paume (+Z) tourne vers le pouce (+X)
  wristDev.rotation.z = -deviationDeg * DEG;   // l'axe de la main (+Y) penche vers le pouce
}

/**
 * Crée une chaîne de phalanges articulées. Renvoie la liste des groupes
 * « articulation » (à faire tourner autour de X), du plus proche au plus loin.
 */
function buildChain(parent, lengths, radius) {
  const joints = [];
  let attach = parent;
  lengths.forEach((len, i) => {
    const joint = new THREE.Group();
    attach.add(joint);
    joints.push(joint);

    // Axe d'articulation (cylindre transversal, comme une goupille).
    const pin = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(radius * 0.55, radius * 0.55, radius * 2.3, 16), pinMat));
    pin.rotation.z = Math.PI / 2;
    joint.add(pin);

    // Phalange : capsule légèrement aplatie, qui s'affine vers le bout.
    const r = radius * (1 - i * 0.08);
    const bone = shadowed(new THREE.Mesh(new THREE.CapsuleGeometry(r, Math.max(len - 2 * r, 0.1), 6, 16), plastic));
    bone.position.y = len / 2;
    bone.scale.z = 0.85;
    joint.add(bone);

    const next = new THREE.Group();
    next.position.y = len;
    joint.add(next);
    attach = next;
  });
  return joints;
}

// Doigts longs : position de la base (x) et longueurs des 3 phalanges.
const LONG_FINGERS = [
  { x: 3.0, lengths: [4.4, 2.8, 2.2], radius: 0.82, tilt: -4 },  // index
  { x: 1.0, lengths: [4.9, 3.1, 2.4], radius: 0.86, tilt: -1 },  // majeur
  { x: -1.0, lengths: [4.6, 2.9, 2.3], radius: 0.82, tilt: 2 },  // annulaire
  { x: -3.0, lengths: [3.6, 2.3, 2.0], radius: 0.72, tilt: 5 },  // auriculaire
];
// Répartition de la flexion (degrés) sur MCP / PIP / DIP pour une commande de 180°.
const LONG_MAX = [80, 95, 70];

const fingers = []; // fingers[i] = fonction(flexDeg) qui pose le doigt i

// --- Pouce : base orientable (opposition) + 2 phalanges.
{
  const base = new THREE.Group();
  // Repère de la paume centré : bas de la paume en y = -palmH / 2.
  base.position.set(palmW / 2 - 0.6, -palmH / 2 + 1.6, 0.3);
  base.rotation.z = -50 * DEG;       // Pouce écarté vers l'extérieur
  palm.add(base);

  const opposition = new THREE.Group();
  base.add(opposition);
  // Métacarpe du pouce (fixe par rapport à la base mobile).
  const meta = shadowed(new THREE.Mesh(new THREE.CapsuleGeometry(1.0, 1.8, 6, 16), plastic));
  meta.position.y = 1.4;
  meta.scale.z = 0.85;
  opposition.add(meta);
  const tip = new THREE.Group();
  tip.position.y = 2.9;
  opposition.add(tip);
  const joints = buildChain(tip, [3.0, 2.5], 0.9);

  fingers.push((flex) => {
    const t = flex / 180;
    // Opposition : le pouce passe devant la paume et pivote vers l'auriculaire.
    opposition.rotation.x = t * 50 * DEG;
    opposition.rotation.z = t * 55 * DEG;
    joints[0].rotation.x = t * 45 * DEG;
    joints[1].rotation.x = t * 75 * DEG;
  });
}

// --- Doigts longs.
for (const f of LONG_FINGERS) {
  const base = new THREE.Group();
  base.position.set(f.x, palmH / 2 - 0.2, 0);
  base.rotation.z = f.tilt * DEG;
  palm.add(base);
  const joints = buildChain(base, f.lengths, f.radius);
  fingers.push((flex) => {
    const t = flex / 180;
    joints.forEach((j, k) => { j.rotation.x = t * LONG_MAX[k] * DEG; });
  });
}

// ------------------------------------------------------- Ligne d'état
const wsDot = document.getElementById("ws-dot"), wsText = document.getElementById("ws-text");
const handDot = document.getElementById("hand-dot"), handText = document.getElementById("hand-text");

// ---------------------------------------------------------- WebSocket
const target = [0, 0, 0, 0, 0];   // flexion reçue (0–180)
const current = [0, 0, 0, 0, 0];  // flexion affichée (interpolée)
const wristTarget = [0, 0];       // rotation, inclinaison reçues (degrés)
const wristCurrent = [0, 0];
const wsPort = new URLSearchParams(location.search).get("ws") || "8765";
const wsUrl = `ws://${location.hostname || "127.0.0.1"}:${wsPort}`;

function connect() {
  const ws = new WebSocket(wsUrl);
  ws.onopen = () => { wsDot.className = "dot ok"; wsText.textContent = `Connecté (${wsUrl})`; };
  ws.onclose = () => {
    wsDot.className = "dot"; wsText.textContent = "Déconnecté — nouvelle tentative…";
    handDot.className = "dot"; handText.textContent = "Main : —";
    setTimeout(connect, 1000);  // reconnexion automatique
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    msg.flex.forEach((v, i) => { target[i] = v; });
    // Poignet (absent ou désactivé → reste en position neutre).
    const wristOn = Array.isArray(msg.wrist_servo) && Array.isArray(msg.wrist);
    msg.wrist?.forEach((v, i) => { wristTarget[i] = wristOn ? v : 0; });
    if (!wristOn) wristTarget[0] = wristTarget[1] = 0;
    handDot.className = msg.hand ? "dot ok" : "dot warn";
    handText.textContent = msg.hand ? "Main détectée" : "Aucune main (retour main ouverte)";
  };
}
connect();

// -------------------------------------------------------------- Boucle
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  // Interpolation exponentielle : animation fluide à 60 FPS même si les
  // données arrivent à ~30 Hz.
  const k = 1 - Math.exp(-clock.getDelta() * 18);
  for (let i = 0; i < 5; i++) {
    current[i] += (target[i] - current[i]) * k;
    fingers[i](current[i]);
  }
  for (let i = 0; i < 2; i++) wristCurrent[i] += (wristTarget[i] - wristCurrent[i]) * k;
  setWrist(wristCurrent[0], wristCurrent[1]);
  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Accès depuis la console du navigateur pour tester :
//   setFlex([0, 180, 90, 0, 0])   setWristTarget(45, -20)
window.setFlex = (arr) => arr.forEach((v, i) => { target[i] = v; });
window.setWristTarget = (rot, dev) => { wristTarget[0] = rot; wristTarget[1] = dev; };
