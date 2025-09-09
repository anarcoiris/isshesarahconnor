# export_traced.py
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()
example = torch.randn(1,3,224,224).to(device)
traced = torch.jit.trace(model, example)
traced.save("models/model_traced.pt")
