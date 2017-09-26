import numpy as np
import pandas as pd
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


birddata = pd.read_csv("~/Code/HarvardX/bird_tracking.csv", encoding='utf-8')
bird_names = pd.unique(birddata.bird_name)
timestamps = []
for k in range(len(birddata)):
    timestamps.append(datetime.datetime.strptime(birddata.date_time.iloc[k][:-3], "%Y-%m-%d %H:%M:%S"))

birddata["timestamp"] = pd.Series(timestamps, index=birddata.index)
times = birddata.timestamp[birddata.bird_name == "Eric"]
data = birddata[birddata.bird_name == "Eric"]
elapsed_time = [time - times[0] for time in times]
elapsed_days = np.array(elapsed_time) / datetime.timedelta(days=1)
next_day = 1
inds = []
daily_mean_speed = []
for (i, t) in enumerate(elapsed_days):
    if t < next_day:
        inds.append(i)
    else:
        daily_mean_speed.append(np.mean(data.speed_2d[inds]))
        next_day += 1
        inds = []
proj = ccrs.Mercator()
plt.figure(figsize=(10, 10))
ax = plt.axes(projection=proj)
ax.set_extent((-25.0, 20.0, 52.0, 10.0))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=":")
for name in bird_names:
    ix = birddata['bird_name'] == name
    x, y = birddata.longitude[ix], birddata.latitude[ix]
    ax.plot(x, y, '.', transform=ccrs.Geodetic(), label=name)
plt.legend(loc="upper left")
# plt.plot(daily_mean_speed)
# plt.xlabel("Day")
#plt.ylabel("Mean speed (m/s)")
# plt.plot(np.array(elapsed_time)/datetime.timedelta(days=1))
# plt.xlabel("Observation")
#plt.ylabel("Elapsed time (days)")
#plt.figure(figsize=(7, 7))
# for names in bird_names:
#    ix = birddata.bird_name == names
#    x, y = birddata.longitude[ix], birddata.latitude[ix]
#plt.plot(x, y, ".", label=names)
#speed = birddata.speed_2d[ix]
#ind = np.isnan(speed)
#plt.hist(speed[~ind], bins=np.linspace(0, 30, 20), normed=True)
#birddata.speed_2d.plot(kind='hist', range=[0, 30])
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
#plt.legend(loc="lower right")
plt.show()
