
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

import plotting as plt

ts = [0.3683784401177368, 0.41582301706264474,
      0.472714202946423, 0.5148895927507579, 0.5450703264672796,
      0.5936794313252106, 0.6374708623486005, 0.6844804419636764,
      0.7297342187600215, 0.7615467858537308, 0.8412488685110593,
      0.9292487974958341, 0.9628480919358906, 1.0337124338165775,
      1.1097566303735233, 1.174570276908002, 1.278895428345114,
      1.3631185535533363, 1.4631499777100552, 1.5374565675993854,
      1.6854705081557355, 1.8088021554530231, 1.9409951859873409,
      2.142101887761452, 2.234399408698845, 2.3801537601744003,
      2.53512974091882, 2.681053140029032, 2.8350611110273776,
      2.9767589243854338, 3.125217151786476, 3.2807104419056787,
      3.4435175511967846, 3.5891012790816688, 3.7404440838645314,
      3.8977249758087886, 4.033457362590802, 4.202088047787148,
      4.288807803814115, 4.497537867786874, 4.652221408405605,
      4.811673490639726, 4.975988071254713, 5.111001929143689,
      5.284295683818412, 5.426604343895581, 5.572223066760438,
      5.721190897651686, 5.873544710577123, 6.068800882277726,
      6.188545890658511, 6.351254606661181, 6.517471668064618,
      6.687220549478125, 6.816862309082978, 6.992838764254372,
      7.127169581183775, 7.30941773404718, 7.448463608541802,
      7.637008379769632, 7.78078051471326, 7.926577808275773,
      8.074398239853334, 8.274633403080427, 8.427159548882658,
      8.581692966928943, 8.738226138725116, 8.896750273359045,
      9.05725531371842, 9.21972994638489, 9.384161615208788,
      9.550536538544957, 9.718839730106055, 9.832105182219715,
      10.003585658215291, 10.176948746127986, 10.293560611726914,
      10.529245021123828, 10.64830506923951, 10.828399117909244,
      11.010277545958493, 11.132509451654728, 11.317308602268037,
      11.503826871932942, 11.629114671623567, 11.818440138644492,
      11.945574580600015, 12.137631967538976, 12.331291181223921,
      12.461273094832437, 12.65753809260907, 12.789230126234553,
      12.988019189460289, 13.121366259539132, 13.32259597122961,
      13.457542005288767, 13.661127677981687, 13.797615896896383]

noisy = [0.00012950204579399138, 0.00355187822220379, -0.003376248776884299,
         -0.00037694775904938274, -0.0009671361635637312, 0.002003082196989566,
         0.003314258599230685, 0.001777393377897473, 0.0005333641360144373,
         0.000565547326511012, -0.0020953260494335074, -0.0049439756648953235,
         -0.0034626859997696187, -0.000901524683105933, -0.001847622399129986,
         -0.0001139905554359817, -0.0010123726003106609, -0.0007891706920360669,
         5.043130235424939e-06, 0.0005730626585100459, 0.0001986033806909452,
         0.00015433257734674184, 0.00020078532276670974, -0.0010644911926763474,
         -0.0025063439488640883, -0.001362497506402822, -0.0015240978097342077,
         -0.0026947390461806737, 0.0010365865945517602, 0.0037562568535372274,
         0.0019167889096301691, 0.00737960455036388, 0.0026294241219382325,
         0.011539054539037424, 0.0021420443576291213, -0.003434272229286017,
         0.00444004925129769, -0.003923953509979419, -0.005868687853030388,
         -0.008125300013093835, -0.009982488536164685, -0.011938458197317878,
         -0.012695046618201667, -0.01263871796733079, -0.01383381682294714,
         -0.01466138075199515, -0.014945387973165513, -0.012754769411936536,
         -0.009408307678624687, -0.003356942675989958, -0.004723243190707638,
         -0.007143609019807031, -0.005923841769241363, -0.005626163673851221,
         -0.006683856742134232, -0.004324622705358413, -0.0039653407385840085,
         -0.005468021538072471, -0.008498487467956062, -0.00998635688281232,
         -0.01002716334451185, -0.011270613626480198, -0.013546605533078846,
         -0.01219238210609252, -0.013884711146758327, -0.014972678133611472,
         -0.015298506962302064, -0.016434018337194066, -0.016229921757881013,
         -0.014000735442195827, -0.013617673884904142, -0.011787466605828096,
         -0.01208056785859168, -0.013319092993536645, -0.013805543702417523,
         -0.014625384314022062, -0.014480234287069849, -0.014596738978515881,
         -0.014929833105476763, -0.014735846522821299, -0.014590243128794167,
         -0.014036364830874823, -0.01549386474546958, -0.01520423690331078,
         -0.015250094265124881, -0.014258095874055, -0.014237702026116038,
         -0.014074376456492099, -0.013100527926511665, -0.013391440112888063,
         -0.013435209249002981, -0.01418156644809823, -0.01432928083885962,
         -0.015215127189815814, -0.014490812430845332, -0.01467233125222195,
         -0.01477613177112657, -0.013940999866912789]

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

savgol = savgol_filter(noisy, 15, 3)
gauss = gaussian_filter1d(noisy, 2)
moving_avg = smooth(noisy, 3)

print(len(noisy), len(savgol), len(gauss), len(moving_avg))

xs = [ts, ts, ts, ts]
ys = [noisy, savgol, gauss, moving_avg]
labels = ['data', 'savgol', 'gauss', 'moving avg']
colors = ['grey', 'k', 'r', 'b']
markers = ['', '', '', '']
styles = ['--', '-', '-', '--']
alphas = [0.5, 1, 1, 0.4]

plt.plot_simple_multi(xs, ys, labels, colors, markers, styles, alphas,
                      xlabel=r'$t$ (Gyr)', ylabel = r'$d^2 {\rm Age}/dr dt$',
                      xmin=0, xmax=14, scale='linear')
