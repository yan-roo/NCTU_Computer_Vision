
function obj_main(P,p_img2,M,tex_name,im_index)

img = imread(tex_name);
img_size = size(img);

%----------------------------------------------------
% mesh-triangulation
%----------------------------------------------------
 tri = delaunay(p_img2(:, 1), p_img2(:, 2)); % 2D delaunay

% trisurf mesh triangulation
 figure(4+im_index);
 trisurf(tri, P(:, 1), P(:, 2), P(:,3));

%----------------------------------------------------
% output .obj file
%----------------------------------------------------
%fid = fopen('model.obj', 'wt');
cmd_fid = ['fid = fopen(''model' num2str(im_index) '.obj'', ''wt'');'];
eval(cmd_fid)
fprintf(fid, '# obj file\n');
cmd_print = ['fprintf(fid, ''mtllib model' num2str(im_index) '.mtl\n\n'');'];
eval(cmd_print);
fprintf(fid, 'usemtl Texture\n');

% output 3D vertex information (3D points)
[len, dummy] = size(P);
for i=1:len
    fprintf(fid, 'v %f %f %f\n', P(i, 1), P(i,2), P(i,3));
end
fprintf(fid, '\n\n\n');

% output vertex texture coordinate (2D points)
for i=1:len
    % texture mapping
    fprintf(fid, 'vt %f %f\n', p_img2(i, 1)/img_size(2), 1-p_img2(i, 2)/img_size(1));
end
fprintf(fid, '\n\n\n');

% output face information
[len_tri, dummy] = size(tri);
bVisible = 0;
for i=1:len_tri
    % 3D mesh normal
    %fprintf('loop %d \n',i);
    bVisible = CheckVisible(M, P(tri(i,1), :)', P(tri(i,2), :)', P(tri(i,3), :)');
    if (bVisible==1)
        fprintf(fid, 'f %d/%d %d/%d %d/%d\n', ...
            tri(i, 1), tri(i, 1), tri(i, 2), tri(i, 2), tri(i, 3), tri(i, 3));
    else
        fprintf(fid, 'f %d/%d %d/%d %d/%d\n', ...
            tri(i, 2), tri(i, 2), tri(i, 1), tri(i, 1), tri(i, 3), tri(i, 3));
    end
end

fclose(fid);

%----------------------------------------------------
% output .mtl file
%----------------------------------------------------
cmd_mtl = ['fid_mtl = fopen(''model' num2str(im_index) '.mtl'', ''wt'');'];
eval(cmd_mtl)
fprintf(fid_mtl, '# MTL file\n');
fprintf(fid_mtl, 'newmtl Texture\n');
fprintf(fid_mtl, 'Ka 1 1 1\nKd 1 1 1\nKs 1 1 1\n');
cmd_fprintf = ['fprintf(fid_mtl, ''map_Kd ' tex_name '\n'');'];
eval(cmd_fprintf)
fclose(fid_mtl);
