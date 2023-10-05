losses = []
valid_dataset_recs = []
valid_dataset_ims = []

for img, lbl in valid_dataset:
  model.eval()
  with torch.no_grad():
    img_new_ax = img.view((1, 1, 28, 28))  # add the batch dim

    rec = model(img_new_ax)
    #print(img.shape, img_new_ax.shape, rec.shape)
    loss_value = loss(rec, img_new_ax).item()
    losses.append(loss_value)
    valid_dataset_recs.append(rec[0].numpy())
    valid_dataset_ims.append(img.numpy())

losses = np.array(losses)
valid_dataset_recs = np.concatenate(valid_dataset_recs)
valid_dataset_ims = np.concatenate(valid_dataset_ims)

m, s = get_mean_std(losses)

cut_off = m+s*3/2

bad = losses > cut_off

print(bad.sum())

plt.hist(losses, 100);
plt.axvline(cut_off)
plt.show()

outliers_ims = valid_dataset_ims[bad]
outliers_recs = valid_dataset_recs[bad]
print(outliers_ims.shape)


i, r = outliers_ims[:32], outliers_recs[:32]
ims = np.stack((i, r), axis=1)
outliers_mosaic = mosaic(ims.reshape(8, 8, 28, 28))
plot_im(outliers_mosaic, is_torch=False)