const { spawn } = require("node:child_process");
const { existsSync } = require("node:fs");
const path = require("node:path");

const root = __dirname;
const backend = path.join(root, "backend");
const frontend = path.join(root, "frontend");
const url = "http://127.0.0.1:3000";
const healthUrl = "http://127.0.0.1:8000/healthz";
const isWindows = process.platform === "win32";
const virtualenvPython = path.join(
  backend,
  isWindows ? ".venv\\Scripts\\python.exe" : ".venv/bin/python",
);
const python = existsSync(virtualenvPython)
  ? { command: virtualenvPython, args: [] }
  : isWindows
    ? { command: "py", args: ["-3"] }
    : { command: "python3", args: [] };

const services = [];
let stopping = false;

function runCommand(name, command, args, cwd, options = {}) {
  const result = spawn(command, args, {
    cwd,
    stdio: "inherit",
    ...options,
  });
  return new Promise((resolve, reject) => {
    result.on("error", (error) => reject(new Error(`${name} could not start: ${error.message}`)));
    result.on("exit", (code, signal) => {
      if (code === 0) resolve();
      else reject(new Error(`${name} failed${signal ? ` (${signal})` : ` with exit code ${code}`}.`));
    });
  });
}

function npmCommand(args) {
  if (process.env.npm_execpath) {
    return {
      command: process.execPath,
      args: [process.env.npm_execpath, ...args],
      options: {},
    };
  }
  return {
    command: isWindows ? "npm.cmd" : "npm",
    args,
    options: isWindows ? { shell: true } : {},
  };
}

async function runNpm(args, cwd = root) {
  const npm = npmCommand(args);
  await runCommand("npm", npm.command, npm.args, cwd, npm.options);
}

async function ensureSetup() {
  const [nodeMajor, nodeMinor] = process.versions.node.split(".").map(Number);
  if (nodeMajor < 20 || (nodeMajor === 20 && nodeMinor < 9)) {
    throw new Error("Node.js 20.9 or newer is required.");
  }

  console.log("Checking Hands-Off OR requirements...");

  if (!existsSync(virtualenvPython)) {
    console.log("Creating the backend Python environment...");
    await runCommand(
      "Python",
      python.command,
      [...python.args, "-m", "venv", path.join(backend, ".venv")],
      root,
    );
  }

  const imports = [
    "cv2",
    "fastapi",
    "mediapipe",
    "numpy",
    "sounddevice",
    "speech_recognition",
    "uvicorn",
    "websockets",
  ].join(", ");
  const importCheck = spawn(
    python.command,
    [...python.args, "-c", `import ${imports}`],
    { cwd: backend, stdio: "ignore" },
  );
  const backendReady = await new Promise((resolve, reject) => {
    importCheck.on("error", (error) => {
      reject(new Error(`Python could not start: ${error.message}`));
    });
    importCheck.on("exit", (code) => resolve(code === 0));
  });

  if (!backendReady) {
    console.log("Installing backend dependencies...");
    await runCommand(
      "Backend dependency installation",
      python.command,
      [...python.args, "-m", "pip", "install", "-r", path.join(backend, "requirements.txt")],
      root,
    );
  }

  const frontendPackages = ["next", "react", "react-dom"];
  const missingFrontendPackages = frontendPackages.filter(
    (name) => !existsSync(path.join(frontend, "node_modules", name, "package.json")),
  );
  if (missingFrontendPackages.length > 0) {
    console.log(
      `Installing missing frontend dependencies: ${missingFrontendPackages.join(", ")}...`,
    );
    await runNpm(["--prefix", frontend, "install", "--no-audit", "--no-fund"]);
  }

  const stillMissingFrontendPackages = frontendPackages.filter(
    (name) => !existsSync(path.join(frontend, "node_modules", name, "package.json")),
  );
  if (stillMissingFrontendPackages.length > 0) {
    throw new Error(
      `Frontend dependencies are still missing: ${stillMissingFrontendPackages.join(", ")}.`,
    );
  }

  if (missingFrontendPackages.length > 0) {
    console.log("Frontend dependencies are ready.");
  }

  if (!existsSync(path.join(frontend, "node_modules", "next", "dist", "bin", "next"))) {
    throw new Error("Next.js could not be found after frontend dependency installation.");
  }

  console.log("Building the optimized frontend...");
  await runNpm(["--prefix", frontend, "run", "build"]);
}

function startService(name, command, args, cwd, { optional = false, ...options } = {}) {
  const service = spawn(command, args, {
    cwd,
    stdio: "inherit",
    ...options,
  });
  services.push({ name, process: service });

  service.on("error", (error) => {
    console.error(`${name} could not start: ${error.message}`);
    if (!optional) stop(1);
  });
  service.on("exit", (code, signal) => {
    if (stopping) return;
    if (optional) {
      console.error(
        `${name} is unavailable${signal ? ` (${signal})` : `; exited with code ${code}`}. The viewer will continue without it.`,
      );
      return;
    }
    console.error(
      `${name} exited unexpectedly${signal ? ` (${signal})` : ` with exit code ${code}`}.`,
    );
    stop(code || 1);
  });
  return service;
}

function stop(exitCode = 0) {
  if (stopping) return;
  stopping = true;
  process.exitCode = exitCode;

  for (const service of services) {
    if (service.process.exitCode === null && service.process.signalCode === null) {
      if (isWindows && service.process.pid != null) {
        const killer = spawn(
          "taskkill.exe",
          ["/PID", String(service.process.pid), "/T", "/F"],
          { stdio: "ignore" },
        );
        killer.on("error", (error) => {
          console.error(`Could not stop ${service.name} process tree: ${error.message}`);
          service.process.kill();
        });
        killer.on("close", (code) => {
          if (
            code !== 0 &&
            service.process.exitCode === null &&
            service.process.signalCode === null
          ) {
            service.process.kill();
          }
        });
      } else {
        service.process.kill();
      }
    }
  }

  setTimeout(() => process.exit(exitCode), 1500).unref();
}

function openBrowser() {
  let browser;
  if (process.platform === "darwin") {
    browser = spawn("open", [url], { stdio: "ignore", detached: true });
  } else if (isWindows) {
    browser = spawn("cmd", ["/c", "start", "", url], { stdio: "ignore", detached: true });
  } else {
    browser = spawn("xdg-open", [url], { stdio: "ignore", detached: true });
  }
  browser.on("error", (error) => {
    console.error(`Could not open a browser: ${error.message}. Open ${url} manually.`);
  });
  browser.unref();
}

async function openWhenReady() {
  const deadline = Date.now() + 120_000;
  while (!stopping && Date.now() < deadline) {
    try {
      const [frontendResponse, backendResponse] = await Promise.all([
        fetch(url, { signal: AbortSignal.timeout(1000) }),
        fetch(healthUrl, { signal: AbortSignal.timeout(1000) }),
      ]);
      const ready = frontendResponse.ok && backendResponse.ok;
      await Promise.all([
        frontendResponse.body?.cancel(),
        backendResponse.body?.cancel(),
      ]);
      if (ready) {
        console.log(`Hands-Off OR is ready at ${url}`);
        openBrowser();
        return;
      }
    } catch {
      // Services are still starting; retry until they respond or the timeout expires.
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  if (!stopping) {
    console.error(`The app did not become ready in time. Check the service output; open ${url} manually.`);
  }
}

process.on("SIGINT", () => stop(0));
process.on("SIGTERM", () => stop(0));

async function main() {
  await ensureSetup();
  if (stopping) return;

  console.log("Starting Hands-Off OR...");
  startService(
    "Backend API",
    python.command,
    [...python.args, "-m", "uvicorn", "server:app", "--port", "8000"],
    backend,
  );
  startService(
    "Vision service",
    python.command,
    [...python.args, "vision.py"],
    backend,
    { optional: true },
  );
  startService(
    "Voice service",
    python.command,
    [...python.args, "voice.py"],
    backend,
    { optional: true },
  );
  startService(
    "Production frontend",
    process.execPath,
    [
      path.join(frontend, "node_modules", "next", "dist", "bin", "next"),
      "start",
      "--hostname",
      "127.0.0.1",
      "--port",
      "3000",
    ],
    frontend,
  );

  void openWhenReady();
}

main().catch((error) => {
  console.error(`Startup failed: ${error.message}`);
  stop(1);
});
