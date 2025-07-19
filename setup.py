from setuptools import setup

setup(
    name="pmt-fitter",
    version="1.1.0",
    description="A PMT charge spectrum fitter using FFT-based convolution",
    author="Yiqi Liu",
    author_email="liuyiqi24@mails.tsinghua.edu.cn",
    py_modules=["pmt_fitter", "tweedie_pdf"],
    install_requires=["numpy", "scipy", "emcee"],
    python_requires=">=3.9",
)
