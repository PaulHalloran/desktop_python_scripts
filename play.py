
plt.close('all')
y = (running_mean_post.running_mean_post(data_final[:,1],36*1))

fig, ax1 = plt.subplots()
ax1.plot(data_final[:,0],y,'k',linewidth=3,alpha=0.5)
ax1.set_ylim([0.0,0.2])
ax2 = ax1.twinx()
ax2.plot(nao[:,0],signal.detrend(nao[:,1]),linewidth=3,alpha=0.5)
plt.xlim([1000,1850])
plt.show(block = False)
