import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import csv
from utils.utils import set_seed
from models import build_model
from datasets import build_dataset
import hydra
from omegaconf import OmegaConf
from utils.visualization import visualize_prediction
import os


def to_device(input, device):
    if isinstance(input, torch.Tensor):
        return input.to(device)
    elif isinstance(input, dict):
        return {k: to_device(v, device) for k, v in input.items()}
    elif isinstance(input, list):
        return [to_device(x, device) for x in input]
    else:
        return input


@hydra.main(version_base=None, config_path="configs", config_name="config")
def generate_predictions(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.ckpt_path)["state_dict"])
    model.to(device)
    model.eval()

    # Load the test dataset
    test_dataset = build_dataset(cfg,val=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=test_dataset.collate_fn)  # Batch size can be adjusted

    os.makedirs("visualizations", exist_ok=True)
    predictions = []
    # Generate predictions
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            batch = to_device(batch, device)
            model_input = {}
            inputs = batch['input_dict']
            agents_in, agents_mask, roads = inputs['obj_trajs'], inputs['obj_trajs_mask'], inputs['map_polylines']
            ego_in = torch.gather(agents_in, 1, inputs['track_index_to_predict'].view(-1, 1, 1, 1).repeat(1, 1, *agents_in.shape[-2:])).squeeze(1)
            ego_mask = torch.gather(agents_mask, 1, inputs['track_index_to_predict'].view(-1, 1, 1).repeat(1, 1,agents_mask.shape[-1])).squeeze(1)
            agents_in = torch.cat([agents_in[..., :2], agents_mask.unsqueeze(-1)], dim=-1)
            agents_in = agents_in.transpose(1, 2)
            ego_in = torch.cat([ego_in[..., :2], ego_mask.unsqueeze(-1)], dim=-1)
            roads = torch.cat([inputs['map_polylines'][..., :2], inputs['map_polylines_mask'].unsqueeze(-1)], dim=-1)
            model_input['ego_in'] = ego_in
            model_input['agents_in'] = agents_in
            model_input['roads'] = roads
            output = model._forward(model_input)
            predictions.extend(output['predicted_trajectory'].cpu().numpy()[..., :2])

            # draw visualizations
            plt = visualize_prediction(batch, output)
            plt.savefig(f"visualizations/visualization_{i}.png")
            plt.close()

    # Save predictions to a csv file for submission
    with open("submission.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Pred_ID', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6'])
        for i, pred in enumerate(predictions):
            Pred_ID = test_dataset.data_loaded[i].split('/')[-1].split('.')[0] + "_pred_"
            for t in range(60):
                Pred_ID_t = Pred_ID + str(t)
                pred_t = pred[:, t, :2]
                writer.writerow([Pred_ID_t, pred_t[0, 0], pred_t[0, 1], pred_t[1, 0], pred_t[1, 1], pred_t[2, 0], pred_t[2, 1], pred_t[3, 0], pred_t[3, 1], pred_t[4, 0], pred_t[4, 1], pred_t[5, 0], pred_t[5, 1]])


if __name__ == "__main__":
    generate_predictions()