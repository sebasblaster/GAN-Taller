# GAN explicada paso a paso

Proyecto educativo para entender los parámetros de una **GAN** (Red Generativa Adversaria):
la red generadora, la red discriminadora y sus funciones de pérdida.

## Contenido

| Archivo | Descripción |
|---|---|
| `gan_explicada.html` | App interactiva (sin instalar nada). Entrena una GAN en el navegador sobre 4 categorías 2D: círculo, dos lunas, espiral y rejilla 3×3. Muestra pérdidas, valores de las redes y el resultado. |
| `gan_mnist.py` | GAN con PyTorch sobre MNIST. La categoría es el dígito elegido. |

## Cómo usar

**App web:** abre `gan_explicada.html` con doble clic en cualquier navegador y pulsa *Entrenar*.

**Versión Python:**

```bash
pip install torch torchvision matplotlib
python gan_mnist.py --digito 7 --epocas 30
```

Genera `resultado_gan.png` con las curvas de pérdida y las imágenes generadas.

## Funciones de pérdida

- Discriminador: `-[ log D(x) + log(1 - D(G(z))) ]`
- Generador: `-log D(G(z))`

## Autor

Tu nombre aquí - Curso / Universidad - Año
