
import torch 

def load_train_model(args, dev):
    from src.models.GATr.Gatr_pf_e_noise import ExampleWrapper as GravnetModel
    model = GravnetModel.load_from_checkpoint(
        args.load_model_weights, args=args, dev=0, map_location=dev,strict=False)
    return model 


def load_test_model(args, dev):
    if args.load_model_weights is not None and (not args.correction):
            from src.models.GATr.Gatr_pf_e_noise import ExampleWrapper as GravnetModel
            model = GravnetModel.load_from_checkpoint(
                args.load_model_weights, args=args, dev=0, map_location=dev, strict=False
            )
           
    if args.load_model_weights is not None and args.correction:
            from src.models.GATr.Gatr_pf_e_noise import ExampleWrapper as GravnetModel
            ckpt = torch.load(args.load_model_weights, map_location=dev)

            state_dict = ckpt["state_dict"]

            # Remove PID head weights
            # keys_to_remove = [
            #     k for k in state_dict
            #     if ((k.startswith("ec_model_wrapper_charged.PID_head"))) #or (k.startswith("ec_model_wrapper_neutral.PID_head")) or (k.startswith("ec_model_wrapper_neutral.gatr_pid")))
            # ] #
           
            # for k in keys_to_remove:
            #     del state_dict[k]
            print("loading state dic clustering")
            model = GravnetModel( args=args, dev=0)
            model.load_state_dict(state_dict, strict=False)

            # model = GravnetModel.load_from_checkpoint(
            #     args.load_model_weights_clustering, args=args, dev=0, strict=False, map_location=torch.device("cuda:2")
            # )
            model2 = GravnetModel.load_from_checkpoint(args.load_model_weights_clustering, args=args, dev=0, strict=False, map_location=torch.device("cuda:1")) # Load the good clustering
            model.gatr = model2.gatr
            model.ScaledGooeyBatchNorm2_1 = model2.ScaledGooeyBatchNorm2_1
            model.clustering = model2.clustering
            model.beta = model2.beta
    return model 



def load_test_model2(args, dev):
    if args.load_model_weights is not None and (not args.correction):
            from src.models.GATr.Gatr_pf_e_noise2 import ExampleWrapper as GravnetModel
            model = GravnetModel.load_from_checkpoint(
                args.load_model_weights, args=args, dev=0, map_location=dev, strict=False
            )
           
    if args.load_model_weights is not None and args.correction:
            from src.models.GATr.Gatr_pf_e_noise import ExampleWrapper as GravnetModel
            ckpt = torch.load(args.load_model_weights, map_location=torch.device("cuda:1"))

            state_dict = ckpt["state_dict"]

            # Remove PID head weights
            # keys_to_remove = [
            #     k for k in state_dict
            #     if ((k.startswith("ec_model_wrapper_charged.PID_head"))) #or (k.startswith("ec_model_wrapper_neutral.PID_head")) or (k.startswith("ec_model_wrapper_neutral.gatr_pid")))
            # ] #
           
            # for k in keys_to_remove:
            #     del state_dict[k]
            model1 = GravnetModel( args=args, dev=0)
        
            model1.load_state_dict(state_dict, strict=False)

            # model = GravnetModel.load_from_checkpoint(
            #     args.load_model_weights_clustering, args=args, dev=0, strict=False, map_location=torch.device("cuda:2")
            # )
            from src.models.GATr.Gatr_pf_e_noise2 import ExampleWrapper 
            model = ExampleWrapper.load_from_checkpoint(args.load_model_weights_clustering, args=args, dev=0, strict=False, map_location=torch.device("cuda:1")) # Load the good clustering
            model.energy_correction = model1.energy_correction
            model.ec_model_wrapper_charged = model1.energy_correction.model_charged
            model.ec_model_wrapper_neutral = model1.energy_correction.model_neutral
    return model 