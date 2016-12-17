
using HDF5
using PyPlot
#using PyPlot
fn = joinpath(homedir(),"Git/FilterFinder/julia/pinky.h5")

f = h5read(fn, "img")
a = f[:,:,1]
b = f[:,:,2]

c = now()
xc = normxcorr2(a[150:200,150:200], b)
d = now()
d-c

imshow(xc, cmap="Greys_r")
imshow(a[1:25, 1:25], cmap="Greys_r")
xcsurface(xc)
i =0

while i <= 1
      a = f[:,:,3]
      b = f[:,:,4]
      xc = normxcorr2(a[50:249, 50:249], b)
      i += 1
end
#fn1 = "/usr/people/davitb/seungmount/research/Alembic/datasets/piriform/3_prealigned/1,3_prealigned.h5"
#fn2 = "/usr/people/davitb/seungmount/research/Alembic/datasets/piriform/3_prealigned/1,4_prealigned.h5"

offset1 = [240, 376]
offset2 = [-375, -317]
s = [1001:2000, 1001:2000]
img1 = h5read(fn1, "img", (s-offset1...))
img2 = h5read(fn2, "img", (s-offset2...))
# block_r
# search_r
