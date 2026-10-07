(() => {
  "use strict";

  const canvas = document.getElementById("viewer");
  const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
  const message = document.getElementById("message");
  if (!gl) {
    message.textContent = "This browser does not provide WebGL, which is required to draw the point cloud.";
    return;
  }

  const vertexSource = `
    attribute vec3 aPosition;
    uniform mat4 uProjection;
    uniform float uYaw;
    uniform float uPitch;
    uniform float uDistance;
    uniform float uPointSize;
    varying float vDepth;

    void main() {
      vec3 p = vec3(aPosition.x, -aPosition.y, 1.55 - aPosition.z);
      float cy = cos(uYaw), sy = sin(uYaw);
      float cp = cos(uPitch), sp = sin(uPitch);
      p = vec3(cy * p.x + sy * p.z, p.y, -sy * p.x + cy * p.z);
      p = vec3(p.x, cp * p.y - sp * p.z, sp * p.y + cp * p.z);
      gl_Position = uProjection * vec4(p.x, p.y, p.z - uDistance, 1.0);
      gl_PointSize = uPointSize;
      vDepth = clamp((aPosition.z - 0.25) / 3.5, 0.0, 1.0);
    }
  `;

  const fragmentSource = `
    precision mediump float;
    varying float vDepth;
    void main() {
      vec2 uv = gl_PointCoord - vec2(0.5);
      if (dot(uv, uv) > 0.25) discard;
      vec3 nearColor = vec3(0.42, 1.0, 0.76);
      vec3 midColor = vec3(0.16, 0.56, 1.0);
      vec3 farColor = vec3(0.68, 0.22, 0.92);
      vec3 color = vDepth < 0.5
        ? mix(nearColor, midColor, vDepth * 2.0)
        : mix(midColor, farColor, (vDepth - 0.5) * 2.0);
      gl_FragColor = vec4(color, 0.95);
    }
  `;

  function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  gl.useProgram(program);

  const positionBuffer = gl.createBuffer();
  const positionLocation = gl.getAttribLocation(program, "aPosition");
  const projectionLocation = gl.getUniformLocation(program, "uProjection");
  const yawLocation = gl.getUniformLocation(program, "uYaw");
  const pitchLocation = gl.getUniformLocation(program, "uPitch");
  const distanceLocation = gl.getUniformLocation(program, "uDistance");
  const pointSizeLocation = gl.getUniformLocation(program, "uPointSize");
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.enableVertexAttribArray(positionLocation);
  gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  let yaw = 0;
  let pitch = 0;
  let distance = 1.8;
  let pointSize = 2;
  let pointCount = 0;
  let paused = false;
  let autoRotate = false;
  let requestPending = false;
  let lastSequence = -1;
  let lastFetchAt = 0;
  let dragging = false;
  let dragX = 0;
  let dragY = 0;

  function perspective(fov, aspect, near, far) {
    const f = 1 / Math.tan(fov / 2);
    const nf = 1 / (near - far);
    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) * nf, -1,
      0, 0, 2 * far * near * nf, 0,
    ]);
  }

  function resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.floor(canvas.clientWidth * ratio);
    const height = Math.floor(canvas.clientHeight * ratio);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      gl.viewport(0, 0, width, height);
      gl.uniformMatrix4fv(projectionLocation, false, perspective(Math.PI / 3, width / height, 0.03, 20));
    }
  }

  async function fetchPoints(now) {
    if (paused || requestPending || now - lastFetchAt < 70) return;
    requestPending = true;
    lastFetchAt = now;
    try {
      const response = await fetch("/api/points", { cache: "no-store" });
      if (!response.ok) throw new Error(`Point stream unavailable (${response.status})`);
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength < 16) throw new Error("Incomplete point frame");
      const view = new DataView(buffer);
      const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
      const version = view.getUint32(4, true);
      const sequence = view.getUint32(8, true);
      const count = view.getUint32(12, true);
      if (magic !== "RSPC" || version !== 1 || buffer.byteLength !== 16 + count * 12) {
        throw new Error("Unsupported point frame format");
      }
      if (sequence !== lastSequence) {
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(buffer, 16, count * 3), gl.DYNAMIC_DRAW);
        pointCount = count;
        lastSequence = sequence;
        document.getElementById("point-count").textContent = count.toLocaleString();
      }
      message.textContent = "";
    } catch (error) {
      message.textContent = error.message;
    } finally {
      requestPending = false;
    }
  }

  async function fetchStatus() {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) throw new Error(`Status unavailable (${response.status})`);
      const status = await response.json();
      document.getElementById("camera-name").textContent = [status.camera, status.serial].filter(Boolean).join(" · ") || "RealSense camera";
      document.getElementById("stream-fps").textContent = status.published_fps ? `${status.published_fps.toFixed(1)} fps` : "—";
      document.getElementById("latency").textContent = status.last_frame_age_ms == null ? "—" : `${Math.round(status.last_frame_age_ms)} ms`;
      const live = document.querySelector(".live");
      const isLive = status.state === "streaming" && status.last_frame_age_ms < 3000;
      live.className = `live ${isLive ? "connected" : status.error ? "error" : ""}`;
      document.getElementById("live-label").textContent = isLive ? "LIVE" : status.state.toUpperCase().replaceAll("_", " ");
      if (status.error) message.textContent = status.error;
    } catch (error) {
      document.querySelector(".live").className = "live error";
      document.getElementById("live-label").textContent = "OFFLINE";
      message.textContent = error.message;
    }
  }

  function render(now) {
    resize();
    if (autoRotate && !dragging) yaw += 0.0025;
    gl.clearColor(0.023, 0.039, 0.047, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.uniform1f(yawLocation, yaw);
    gl.uniform1f(pitchLocation, pitch);
    gl.uniform1f(distanceLocation, distance);
    gl.uniform1f(pointSizeLocation, pointSize * Math.min(window.devicePixelRatio || 1, 2));
    gl.drawArrays(gl.POINTS, 0, pointCount);
    fetchPoints(now);
    requestAnimationFrame(render);
  }

  function resetView() {
    yaw = 0;
    pitch = 0;
    distance = 1.8;
  }

  canvas.addEventListener("pointerdown", (event) => {
    dragging = true;
    dragX = event.clientX;
    dragY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    yaw += (event.clientX - dragX) * 0.006;
    pitch = Math.max(-1.45, Math.min(1.45, pitch + (event.clientY - dragY) * 0.006));
    dragX = event.clientX;
    dragY = event.clientY;
  });
  canvas.addEventListener("pointerup", () => { dragging = false; });
  canvas.addEventListener("pointercancel", () => { dragging = false; });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    distance = Math.max(0.35, Math.min(7, distance * Math.exp(event.deltaY * 0.001)));
  }, { passive: false });
  canvas.addEventListener("dblclick", resetView);

  document.getElementById("reset-view").addEventListener("click", resetView);
  document.getElementById("point-size").addEventListener("input", (event) => {
    pointSize = Number(event.target.value);
    document.getElementById("point-size-value").textContent = pointSize.toFixed(1);
  });
  document.getElementById("auto-rotate").addEventListener("change", (event) => {
    autoRotate = event.target.checked;
  });
  document.getElementById("pause-stream").addEventListener("click", (event) => {
    paused = !paused;
    event.target.textContent = paused ? "Resume" : "Pause";
  });

  fetchStatus();
  setInterval(fetchStatus, 1000);
  requestAnimationFrame(render);
})();
