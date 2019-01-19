1. 创建新环境
```
  conda create --name <env_name> <package_name>
```
- 注意：
  - <env_name> 即创建的环境名。建议以英文命名，且不加空格，名称两边不加尖括号“<>”。
  - <package_names> 即安装在环境中的包名。名称两边不加尖括号“<>”。
  - **conda create --name Helab python=3.6 numpy pandas matplotlib**

2. 删除环境
```
  conda remove --name <env_name> <package_name>
```
