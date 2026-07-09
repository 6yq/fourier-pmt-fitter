from setuptools import setup

setup(
    name="fourier-fitter",
    version="2.0.0",
    description="JAX-based PMT charge spectrum fitter with Fourier-domain models",
    author="Yiqi Liu",
    author_email="liuyiqi24@mails.tsinghua.edu.cn",
    packages=[
        "fourier_fitter",
        "fourier_fitter.core",
        "fourier_fitter.models",
    ],
    package_dir={"fourier_fitter": "."},
    install_requires=["numpy", "scipy", "jax"],
    extras_require={"unbinned": ["jax-finufft"]},
    python_requires=">=3.10",
)
