function [ bVisible ] = CheckVisible( M, P1, P2, P3 )
% used to check if the surface normal facing the camera
%  M: 3x4 projection matrix
%  P1, P2, P3: 3D points

tri_normal = cross((P2-P1), (P3-P2)); % surface normal of mesh

cam_dir = [M(3, 1); M(3, 2); M(3, 3);]; % camera direction

if (dot(cam_dir, tri_normal)<0)
    bVisible = 1;  % visible
    %fprintf('Visible!\n');
else
    bVisible = 0;  % invisible
    %fprintf('inVisible!\n');
end
