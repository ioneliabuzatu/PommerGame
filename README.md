# Play Pommerman with A2C agent

[Competition presentation slides](https://docs.google.
com/presentation/d/1xzbDiAdsOS4i5DYl986K1Cy4fntX2Z3lowpsiPS202w/edit?usp=sharing)

Three games of pommerman

![games](https://imgur.com/GmthDXw.gif)

### install pommerman env
```
git clone https://github.com/MultiAgentLearning/playground ./pommer_setup
pip install -U ./pommer_setup
rm -rf ./pommer_setup

git clone https://github.com/RLCommunity/graphic_pomme_env ./graphic_pomme_env
pip install -U ./graphic_pomme_env
rm -rf ./graphic_pomme_env
```

### Install requirements
`pip install -r requirements.txt`

### Other dependencies
- maybe you need to install torch - pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

#### Training reward 
![](assets/pommerman_reward.png)


### TODO: Reward Redistirbution

References:

Initial game setup https://github.com/plassma/Pommerman-DRL
Model https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail


