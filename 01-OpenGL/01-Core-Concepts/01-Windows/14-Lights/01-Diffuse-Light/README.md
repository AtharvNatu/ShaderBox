STEPS FOR DIFFUSE LIGHT
------------------------------------------------------------------------------------------------------------------------------------------------------
1. Calculate eye co-ordinates by multiplying with modelView matrix

2. Calculate normal matrix from modelView matrix

3. Calculate transformed normals by multiplying normal with above normal matrix.

4. Calculate the source light direction by subtracting eye co-ordinates from light position.

5. Calculate the diffused light by using the following equation - 

    L = Ld * Kd * (s.n)

    Ld -> Value given to diffused light by us
    Kd -> Material diffuse constant
    s -> Source light direction
    n -> Transformed normals
    . -> Dot product
------------------------------------------------------------------------------------------------------------------------------------------------------
