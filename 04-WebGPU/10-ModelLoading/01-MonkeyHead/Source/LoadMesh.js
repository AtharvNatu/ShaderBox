async function loadMesh(_objFilePath)
{
    const response = await fetch(_objFilePath);
    if (!response.ok)
       throw Error(`Failed To Load Mesh ${_objFilePath} : ${response.status}`);

    const modelObj = await response.text();

    const parsedObj = parseObj(modelObj);
    const gpuMesh = buildGPUMesh(parsedObj);

    return gpuMesh;
}

// Obj Parsing
function parseObj(_objData)
{
    // Code
    const mesh = 
    {
        positions: [],
        texcoords: [],
        normals: [],

        positionIndices: [],
        texcoordIndices: [],
        normalIndices: [],
    };

    const lines = _objData.split(/\r?\n/);

    for (let line of lines)
    {
        line = line.trim();

        if (!line || line.startsWith("#"))
            continue;

        const tokens = line.split(/\s+/);
        const type = tokens[0];

        // v: x y z
        if (type == "v")
        {
            mesh.positions.push(
                Number(tokens[1]),
                Number(tokens[2]),
                Number(tokens[3])
            );
        }

        // vt: u v
        else if (type == "vt")
        {
            mesh.texcoords.push(
                Number(tokens[1]),
                Number(tokens[2])
            );
        }

        // vn: x y z
        else if (type == "vn")
        {
            mesh.normals.push(
                Number(tokens[1]),
                Number(tokens[2]),
                Number(tokens[3])
            );
        }

        // f : v/vt/vn ...
        else if (type == "f")
        {
            const face = [];

            for (let i = 1; i < tokens.length; i++)
            {
                face.push(
                    parseFaceVertex(
                        tokens[i],
                        mesh.positions.length / 3,
                        mesh.texcoords.length / 2,
                        mesh.normals.length / 3
                    )
                );
            }

            // OBJ faces can contain more than 3 vertices
            for (let i = 1; i < face.length - 1; i++)
            {
                pushFaceVertex(mesh, face[0]);
                pushFaceVertex(mesh, face[i]);
                pushFaceVertex(mesh, face[i + 1]);
            }
        }
    }

    return mesh;
}

// Parse Face Vertices : v/vt/vn
function parseFaceVertex(_spec, _positionCount, _texcoordCount, _normalCount)
{
    // Code
    const parts = _spec.split("/");

    return {
        positionIndex: getObjIndex(parts[0], _positionCount),
        texcoordIndex: parts[1] && parts[1].length > 0 ? getObjIndex(parts[1], _texcoordCount) : -1,
        normalIndex: parts[2] && parts[2].length > 0 ? getObjIndex(parts[2], _normalCount) : -1
    };
}

function getObjIndex(_value, _count)
{
    // Code
    const index = Number(_value);

    if (index > 0)
        return index - 1;

    if (index < 0)
        return _count + index;

    return -1;
}

function pushFaceVertex(_mesh, _vertex)
{
    // Code
    _mesh.positionIndices.push(_vertex.positionIndex);
    _mesh.texcoordIndices.push(_vertex.texcoordIndex);
    _mesh.normalIndices.push(_vertex.normalIndex);
}

// Convert OBJ to Indexed Drawing Representation
function buildGPUMesh(_obj)
{
    // Code
    const vertices = [];
    const indices = [];

    const vertexMap = new Map();
    const count = _obj.positionIndices.length;

    for (let i = 0; i < count; i++)
    {
        const positionIndex = _obj.positionIndices[i];
        const texcoordIndex = _obj.texcoordIndices[i];
        const normalIndex = _obj.normalIndices[i];

        const key = 
            `${positionIndex}/` +
            `${texcoordIndex}/` +
            `${normalIndex}`;

        let gpuIndex = vertexMap.get(key);

        if (gpuIndex == undefined)
        {
            gpuIndex = vertices.length / 8;

            vertexMap.set(key, gpuIndex);

            // Position
            const p = positionIndex * 3;
            const px = _obj.positions[p + 0];
            const py = _obj.positions[p + 1];
            const pz = _obj.positions[p + 2];

            // Texcoords
            let u = 0;
            let v = 0;
            if (texcoordIndex >= 0)
            {
                const t = texcoordIndex * 2;
                u = _obj.texcoords[t + 0];
                v = _obj.texcoords[t + 1];
                v = 1 - v;
            }

            // Normals
            let nx = 0;
            let ny = 0;
            let nz = 0;
            if (normalIndex >= 0)
            {
                const n = normalIndex * 3;
                nx = _obj.normals[n + 0];
                ny = _obj.normals[n + 1];
                nz = _obj.normals[n + 2];
            }

            // Interleaved GPU Vertex
            vertices.push(
                px,
                py,
                pz,

                u,
                v,

                nx,
                ny,
                nz
            );
        }

        indices.push(gpuIndex);
    }

    return {
        vertices: new Float32Array(vertices),
        indices: new Uint32Array(indices),
        vertexCount: vertices.length / 8,
        indexCount: indices.length
    };
}
