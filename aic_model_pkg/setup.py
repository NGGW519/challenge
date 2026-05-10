from setuptools import find_packages, setup

package_name = "aic_model_pkg"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        # weights는 패키지 share에 함께 install되도록 (런타임 ament_index로 찾기)
        (f"share/{package_name}/weights", []),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    include_package_data=True,
    package_data={
        package_name: ["weights/*.pt", "weights/*.pth", "weights/*.json"],
    },
    maintainer="nggw519",
    maintainer_email="srrd1357@gmail.com",
    description="AIC Challenge participant policy (HybridPolicy).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],  # aic_model의 dynamic loader가 클래스를 찾으므로 별도 entrypoint 불필요
    },
)
