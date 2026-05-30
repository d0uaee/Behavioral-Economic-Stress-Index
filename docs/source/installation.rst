Installation
===========

Prerequis
---------

- Python 3.10 ou superieur
- Git
- 4 Go de RAM minimum (8 Go recommandes pour le GridSearch LSTM)

Cloner le depot
---------------

.. code-block:: bash

   git clone https://github.com/d0uaee/Behavioral-Economic-Stress-Index.git
   cd Behavioral-Economic-Stress-Index

Creer un environnement virtuel
------------------------------

.. code-block:: bash

   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux / macOS
   source venv/bin/activate

Installer les dependances
-------------------------

.. code-block:: bash

   pip install -r requirements.txt

Contenu du fichier ``requirements.txt`` :

.. code-block:: text

   pandas>=2.0
   numpy>=2.0
   matplotlib>=3.7
   seaborn>=0.12
   statsmodels>=0.14
   scikit-learn>=1.3
   scipy>=1.10
   pytrends>=4.9
   tensorflow>=2.13
   prophet
   requests
   beautifulsoup4
   lxml
   jupyter
   nbformat

Verifier l'installation
-----------------------

.. code-block:: bash

   python -c "import pandas, numpy, statsmodels, tensorflow; print('OK')"

Structure attendue apres installation
--------------------------------------

.. code-block:: text

   Behavioral-Economic-Stress-Index/
   |-- data/
   |   |-- bronze/
   |   |-- silver/
   |   `-- gold/
   |       `-- model_dataset_monthly.csv   <- Dataset principal (96 mois x 45 col.)
   |-- src/
   |-- notebooks/
   |-- outputs/
   |-- requirements.txt
   |-- run_v3.py
   `-- README.md
