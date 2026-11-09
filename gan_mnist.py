"""
GAN explicada con PyTorch y MNIST
---------------------------------
Uso:
    pip install torch torchvision matplotlib
    python gan_mnist.py --digito 7 --epocas 30

La "categoría" es el dígito (0-9) que la GAN aprende a generar.
"""
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ------------------------------------------------------------------
# 1) HIPERPARÁMETROS
# ------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--digito", type=int, default=7)   # categoría usada
ap.add_argument("--epocas", type=int, default=30)
ap.add_argument("--lr", type=float, default=2e-4)  # tasa de aprendizaje
ap.add_argument("--batch", type=int, default=64)
args = ap.parse_args()

LATENT = 64                      # tamaño del ruido z
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
# 2) DATOS REALES: solo imágenes del dígito elegido, escaladas a [-1, 1]
# ------------------------------------------------------------------
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist = datasets.MNIST("data", train=True, download=True, transform=tf)
idx = (mnist.targets == args.digito).nonzero().squeeze()
loader = DataLoader(torch.utils.data.Subset(mnist, idx), batch_size=args.batch,
                    shuffle=True, drop_last=True)

# ------------------------------------------------------------------
# 3) AQUÍ SE CREA LA RED GENERADORA
#    Entrada: ruido z (64 números) -> Salida: imagen 28x28 falsa
# ------------------------------------------------------------------
G = nn.Sequential(
    nn.Linear(LATENT, 256), nn.LeakyReLU(0.2),   # capa oculta 1
    nn.Linear(256, 512),    nn.LeakyReLU(0.2),   # capa oculta 2
    nn.Linear(512, 784),    nn.Tanh(),           # salida: 784 píxeles en [-1, 1]
).to(dev)

# ------------------------------------------------------------------
# 4) AQUÍ SE CREA LA RED DISCRIMINADORA
#    Entrada: imagen (784 píxeles) -> Salida: probabilidad de ser real
# ------------------------------------------------------------------
D = nn.Sequential(
    nn.Linear(784, 512), nn.LeakyReLU(0.2),      # capa oculta 1
    nn.Linear(512, 256), nn.LeakyReLU(0.2),      # capa oculta 2
    nn.Linear(256, 1),   nn.Sigmoid(),           # salida: probabilidad (0 = falso, 1 = real)
).to(dev)

# ------------------------------------------------------------------
# 5) FUNCIÓN DE PÉRDIDA Y OPTIMIZADORES
#    BCELoss = -[y*log(p) + (1-y)*log(1-p)]  (entropía cruzada binaria)
# ------------------------------------------------------------------
loss_fn = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))


def resumen(nombre, red):
    """Imprime los valores (parámetros) de cada capa lineal de una red."""
    print(f"\n--- {nombre} ---")
    total = 0
    for i, capa in enumerate(m for m in red if isinstance(m, nn.Linear)):
        w, b = capa.weight.data, capa.bias.data
        total += w.numel() + b.numel()
        print(f"Capa {i+1}: {capa.in_features}->{capa.out_features} | "
              f"peso medio={w.mean():+.4f} desv={w.std():.4f} | sesgo medio={b.mean():+.4f}")
    print(f"Total de parámetros: {total:,}")


resumen("GENERADOR (antes de entrenar)", G)
resumen("DISCRIMINADOR (antes de entrenar)", D)

# ------------------------------------------------------------------
# 6) ENTRENAMIENTO
# ------------------------------------------------------------------
hist_D, hist_G = [], []
for ep in range(args.epocas):
    for real, _ in loader:
        real = real.view(args.batch, -1).to(dev)
        unos = torch.ones(args.batch, 1, device=dev)    # etiqueta "real"
        ceros = torch.zeros(args.batch, 1, device=dev)  # etiqueta "falso"

        # (a) Entrenar D: debe decir 1 a lo real y 0 a lo falso
        z = torch.randn(args.batch, LATENT, device=dev)
        falso = G(z).detach()                           # detach: aquí G no se actualiza
        loss_D = loss_fn(D(real), unos) + loss_fn(D(falso), ceros)
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # (b) Entrenar G: debe lograr que D diga 1 a sus imágenes falsas
        z = torch.randn(args.batch, LATENT, device=dev)
        loss_G = loss_fn(D(G(z)), unos)                 # pérdida no saturada: -log D(G(z))
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

    hist_D.append(loss_D.item()); hist_G.append(loss_G.item())
    print(f"Época {ep+1}/{args.epocas} | pérdida D={loss_D.item():.3f} | pérdida G={loss_G.item():.3f}")

resumen("GENERADOR (después de entrenar)", G)
resumen("DISCRIMINADOR (después de entrenar)", D)

# ------------------------------------------------------------------
# 7) RESULTADOS: curvas de pérdida + imágenes generadas
# ------------------------------------------------------------------
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(hist_D, label="Pérdida D"); ax.plot(hist_G, label="Pérdida G")
ax.set_title("Funciones de pérdida"); ax.set_xlabel("Época"); ax.legend()

with torch.no_grad():
    imgs = G(torch.randn(16, LATENT, device=dev)).view(16, 28, 28).cpu()
cuadro = torch.cat([torch.cat(list(imgs[r*4:(r+1)*4]), dim=1) for r in range(4)], dim=0)
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(cuadro, cmap="gray"); ax2.axis("off")
ax2.set_title(f"Resultado de la GAN - categoría: dígito {args.digito}")
plt.tight_layout(); plt.savefig("resultado_gan.png", dpi=120); plt.show()
