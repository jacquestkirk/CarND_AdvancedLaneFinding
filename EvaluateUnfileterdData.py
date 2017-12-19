import pickle

[leftLine, rightLine, time] =pickle.load( open( "UnFiltered.p", "rb" ) )

time = np.linspace(0, (end_time - start_time), leftLine.center_distance.__len__())

fig1, (ax_fit0, ax_fit1, ax_fit2, ax_roc, ax_center) = plt.subplots(5, 1, sharex=True)


ax_fit0.plot(time, leftLine.line_history[:, 0], color='r')
ax_fit0.plot(time, rightLine.line_history[:, 0], color='b')
ax_fit0.plot(time, np.abs(leftLine.line_history[:, 0] - rightLine.line_history[:, 0]), color='g')
ax_fit0.set_ylabel('fit0')

ax_fit1.plot(time, leftLine.line_history[:, 1], color='r')
ax_fit1.plot(time, rightLine.line_history[:, 1], color='b')
ax_fit1.plot(time, np.abs(leftLine.line_history[:, 1] - rightLine.line_history[:, 1]), color='g')
ax_fit1.set_ylabel('fit1')

ax_fit2.plot(time, leftLine.line_history[:, 2], color='r')
ax_fit2.plot(time, rightLine.line_history[:, 2], color='b')
ax_fit2.set_ylabel('fit2')

ax_roc.plot(time, leftLine.radius_of_curvature, color='r')
ax_roc.plot(time, rightLine.radius_of_curvature, color='b')
ax_roc.set_ylabel('Radius Of Curvature')

ax_center.plot(time, leftLine.center_distance, color='b')
ax_center.set_ylabel('Center')

