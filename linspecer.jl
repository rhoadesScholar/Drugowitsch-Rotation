using SimplePCHIP

# function lineStyles = linspecer(N)
# This function creates an Nx3 array of N [R B G] colors
# These can be used to plot lots of lines with distinguishable & nice
# looking colors.
#
# lineStyles = linspecer(N);  makes N colors for you to use: lineStyles[ii,:]
#
# colormap(linspecer); set your colormap to have easily distinguishable
#                      colors & a pleasing aesthetic
#
# lineStyles = linspecer(N,"qualitative"); forces the colors to all be distinguishable [up to 12]
# lineStyles = linspecer(N,"sequential"); forces the colors to vary along a spectrum
#

function linspecer(n=128)

    if n<=0 # its empty; nothing else to do here
        lineStyles=[]
        return
    end

    if n==1
        return [0.2005 0.5593 0.7380]
    end
    if n==2
        return  [0.2005 0.5593 0.7380; 0.9684 0.4799 0.2723]
    end

    frac=.85; # Slight modification from colorbrewer here to make the yellows in the center just a bit darker
    cmapp = [158 1 66; 213 62 79; 244 109 67; 253 174 97; 254 224 139; 255*frac 255*frac 191*frac; 230 245 152; 171 221 164; 102 194 165; 50 136 189; 94 79 162]
    x = range(1, n, length=size(cmapp,1))
    xi = 1:n
    cmap = zeros(n,3)
    for ii=1:3
        cmap[:,ii] = [SimplePCHIP.interpolate(x,cmapp[:,ii])(xii) for xii in xi]
    end
    cmap = reverse(cmap/255, dims = 1)

    return cmap
end
