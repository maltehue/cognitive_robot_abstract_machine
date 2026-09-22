/* ============================================================================
 * teleop_hand.mjs — webcam hand tracking for the Teleoperation page.
 *
 * Loads MediaPipe's HandLandmarker (vendored locally under vendor/mediapipe),
 * runs it on the Mac's webcam, draws the landmarks over a mirrored preview, and
 * hands each frame's wrist position and pinch amount to teleop.js via the global
 * window.teleopOnHand callback. All state lives here; teleop.js only consumes.
 * ==========================================================================*/
import {
  FilesetResolver,
  HandLandmarker,
} from './vendor/mediapipe/vision_bundle.mjs';

const WASM_PATH = 'vendor/mediapipe/wasm';
const MODEL_PATH = 'vendor/mediapipe/hand_landmarker.task';

// the finger bones, as index pairs into the 21 hand landmarks, for the overlay
const BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],          // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],          // index
  [5, 9], [9, 10], [10, 11], [11, 12],     // middle
  [9, 13], [13, 14], [14, 15], [15, 16],   // ring
  [13, 17], [17, 18], [18, 19], [19, 20],  // pinky
  [0, 17],                                 // palm base
];

let landmarker = null;
let stream = null;
let running = false;

async function ensureLandmarker() {
  if (landmarker) return landmarker;
  const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
  landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_PATH },
    numHands: 1,
    runningMode: 'VIDEO',
  });
  return landmarker;
}

function pinchAmount(hand) {
  // thumb tip (4) to index tip (8), normalised by hand span (wrist 0 to middle MCP 9),
  // so it is roughly scale-invariant: ~0 when pinched shut, ~1+ when open
  const d = (a, b) => Math.hypot(hand[a].x - hand[b].x, hand[a].y - hand[b].y);
  const span = d(0, 9) || 1e-3;
  return d(4, 8) / span;
}

function draw(canvas, video, hands) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.save();
  ctx.setTransform(-1, 0, 0, 1, w, 0); // mirror: a selfie view reads naturally
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();
  if (!hands.length) return;
  const hand = hands[0];
  ctx.save();
  ctx.setTransform(-1, 0, 0, 1, w, 0);
  ctx.strokeStyle = 'rgba(47,125,209,0.9)';
  ctx.lineWidth = 3;
  for (const [a, b] of BONES) {
    ctx.beginPath();
    ctx.moveTo(hand[a].x * w, hand[a].y * h);
    ctx.lineTo(hand[b].x * w, hand[b].y * h);
    ctx.stroke();
  }
  ctx.fillStyle = '#8fd6c8';
  for (const p of hand) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = '#ffcf5b'; // wrist, the point that drives the arm
  ctx.beginPath();
  ctx.arc(hand[0].x * w, hand[0].y * h, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function loop(video, canvas) {
  if (!running) return;
  let hands = [];
  try {
    const result = landmarker.detectForVideo(video, performance.now());
    hands = result.landmarks || [];
  } catch (_) { /* a dropped frame must not stop the loop */ }
  draw(canvas, video, hands);
  if (hands.length) {
    const hand = hands[0];
    // wrist in [0,1] image coords, plus the apparent hand size as a depth proxy (a bigger
    // hand is nearer the camera) and the pinch amount; teleop.js maps these onto the arm
    const span = Math.hypot(hand[0].x - hand[9].x, hand[0].y - hand[9].y);
    window.teleopOnHand({
      present: true,
      x: hand[0].x,
      y: hand[0].y,
      span: span,
      pinch: pinchAmount(hand),
    });
  } else {
    window.teleopOnHand({ present: false });
  }
  requestAnimationFrame(() => loop(video, canvas));
}

async function start(video, canvas) {
  await ensureLandmarker();
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: 'user' },
  });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  running = true;
  loop(video, canvas);
}

function stop() {
  running = false;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
}

window.teleopHand = { start, stop };
