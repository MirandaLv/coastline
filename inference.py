
import os
import torch
import rasterio

def run_inference(model, dataloader, output_dir, device):

    model.eval()
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        print("Working on Inferencing...")
        for i, batch in enumerate(dataloader):

            print("Working on batch {}".format(i))
            X = batch['image'].to(device)
            filenames = batch['filename']  # list of filenames in the batch
            output = model(X)['out']
            preds = torch.argmax(output, dim=1).cpu().numpy()  # shape: (B, H, W)

            for pred, fname in zip(preds, filenames):
                original_path = os.path.join(dataloader.dataset.folder_path, fname)
                with rasterio.open(original_path) as src:
                    profile = src.profile
                    profile.update({
                        "count": 1,
                        "dtype": 'uint8',
                        "compress": "lzw"})

                    out_path = os.path.join(output_dir, f"pred_{fname}")
                    with rasterio.open(out_path, 'w', **profile) as dst:
                        dst.write(pred.astype('uint8'), 1)  # Write to band 1

    print(f"Inference complete. Predictions saved to: {output_dir}")



