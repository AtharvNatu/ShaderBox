# WebGPU Samples

## Requires a WebGPU supported browser
    - Chrome/Edge 113+
    - Safari 26+
    - Firefox 141+

## Instructions

- Use HTTP Server - shaders, textures and models are loaded with fetch():

    - Run with `Live Server` extension or inbuilt server in VS Code. Additionally, Python HTTP Server for Texture based samples

    OR

    ```sh
    python3 -m http.server 8080 --directory 02-WebGPU
    ```

    then open `http://localhost:8080/<sample>/Source/Canvas.html`

- Keys:

    - F       : Fullscreen
    - Q       : Quit
    - L       : Lighting (09)
    - V/F     : Per-Vertex / Per-Fragment (09-05)
    - X/Y/Z   : Light Axis (09-08)
    - 1-4     : Texcoord Modes (08-02)
    - T       : CPU/GPU (13)
    - 1-5     : Mesh Size (13)
