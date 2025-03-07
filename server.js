const express = require('express'); //para manejar peticiones http
const axios = require('axios'); //nos permite hacer las solicitudes http
const tf = require('@tensorflow/tfjs'); //para trabajar con modelos d aprendizaje profundo en JS
const {Image, createCanvas} = require('canvas'); //nos permite manipular imgs en Node.js
const fs = require('fs').promises; //version asincrona de fs para manejar archivos
const path = require('path'); //modulo para manejar rutas d archivos
const multer = require('multer'); //para manejar la subida d archivos
const { stat } = require('fs');

//creamos instancia d app express y definimos el puerto
const app = express();
const PORT = 3000;

//habilitamos el manejo d json en las solicitudes
app.use(express.json());

//definimos las rutas dnd guardamos el modelo y sus datos d pesos
const modelPath = path.join(__dirname, 'model.json');
const dataPath = path.join(__dirname, 'model-data.json');

//urls de la api randomuser para obtener las imgs de mujeres y hombres
const maleUrl = 'https://randomuser.me/api/?gender=male&results=50';
const femaleUrl = 'https://randomuser.me/api/?gender=female&results=50';

//funcion para obtener datos d la api y estructurarlos
async function fetchData(url) {
    const response = await axios.get(url); //hacemos la solicitud http con axios
    const data = response.data; //extraemos los datos d la respuesta
    return data.results.map( user => ({
        image: user.picture.medium, //url d la img del user
        label: user.gender === 'male' ? 0 : 1 //asignamos la etiqueta en base al genero (0 para hombres y 1 para mujeres)
    }));
}

//funcion q carga y procesa las imgs para entrenar el modelo
async function loadAndProcessImgs(data) {
    const imgs = [];
    const labels = [];

    for(const item of data) {
        try {
            const response = await axios.get(item.image, {responseType: 'arraybuffer'});
            const buffer = Buffer.from(response.data); //convertimos los datos en un buffer d imagen

            //creamos una img en memoria desde el buffer
            const img = new Image();
            img.src = buffer;

            //creamos un canvas y dibujamos la img redimensionada a 72x72 pixeles
            const canvas = createCanvas(72, 72);
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, 72, 72);

            //convertimos la img en un tensor y normalizamos los valores (entre 1 y 0)
            let imgTensor = tf.browser.fromPixels(canvas)
                .toFloat()
                .div(255);

            imgs.push(imgTensor);
            labels.push(item.label);

        } catch(err) {
            console.error('Error al procesar la imagen: ', item.image, err);
        }
    }
    return {
        xs: tf.stack(imgs).reshape([imgs.length, 72, 72, 3]), //agrupamos las imgs en un solo tensor
        ys: tf.tensor(labels, [labels.length, 1]) //creamos el tensor d etiquetas
    };
}

//creacion del modelo d red neuronal convolucional
function createModel() {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({inputShape: [72, 72, 3], filters: 16, kernelSize: 3, activation: 'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize: 2}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({units: 10, activation: 'relu'}));
    model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

    model.compile({optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy']});
    return model;
}

let savedModelJSON = null;
let savedModelWeights = null;

async function saveModel(model) {
    console.log('--- Guardando modelo ---');
    const modelJSON = model.toJSON();
    //por estandar siempre debe ir en el prop modelTopology
    const modelData = { modelTopology: JSON.parse(modelJSON)};

    //lo almacenamos en el fichero
    await fs.writeFile(modelPath, JSON.stringify(modelData, null, 2));
    
    //guardamos los pesos en formato json
    const weights = model.getWeights();
    //Devolvemos un vector con objetos q contienen los datos y la forma d los tensores del modelo
    const weightsData = await Promise.all(weights.map( async (w) => ({
        data: Array.from(await w.data()), // extrae los valores numericos
        shape: w.shape //guarda la forma del tensor
    })));

    await fs.writeFile(dataPath, JSON.stringify(weightsData, null, 2));

    console.log('--- Modelo guardado con éxito ---');
}

//funcion q verifica si el modelo y los pesos existen en el sistema d archivos
async function modelExists() {
    try {
        //**Guarrada:
        //intentamos acceder a los archivos del modelo y los pesos
        await fs.access(modelPath);
        await fs.access(dataPath);
        return true; //si ambos existen asumimos q estan almacenados
    } catch {
        return false;
    }
}

async function loadModel() {
    try {
        console.log('--> Cargando modelo almacenado...');

        //lee el archivo json q contiene la estructura del modelo
        const modelJsonRaw = await fs.readFile(modelPath, 'utf8');
        const modelJSON = JSON.parse(modelJsonRaw); //convierte el json en un obj de js

        //verificamos si el modelo contiene los datos (q deben estar estar en la prop modelTopology)
        if( !modelJSON.modelTopology) {
            throw new Error('El modelo guardado tiene un formato incorrecto.');
        }

        //cargamos el modelo dsd el json q contiene su topologia
        const model = await tf.models.modelFromJSON(modelJSON.modelTopology);

        //leemos el archivo d pesos y reconstruimos los tensores con su forma original
        const weightsData = JSON.parse( await fs.readFile(tf.dataPath, 'utf8'));

        //'iteramos' sobre los datos d los pesos y reconstruimos el tensor
        const weights = weightsData.map( w => tf.tensor(w.data, w.shape));

        model.setWeights(weights); //asignamos los pesos al modelo

        console.log('--> Modelo cargado con éxito.');
        return model;
    } catch (err){
        console.log('No se ha encontrado modelo almacenado o ha sucedido un error al cargarlo.');
        console.log('--- Creando un modelo nuevo ---');
        return createModel(); //si hay error creamos un modelo nuevo
    }
}

//función para entrenar el modelo con los datos ya procesados
async function trainModel(model, trainData) {
    console.log('--- Empieza entrenamiento ---');

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 2, //num d épocas de entrenamiento
        batchSize: 64, //tamaño del lote
        shuffle: true, //mezclar los datos para mejorar el entrenamiento
        validationSplit: 0.2 //porcentaje d datos utilizados para la validación
    });

    console.log('--- Entrenamiento completado ---');

    //** guardamos el modelo en memoria / variable para printearlo en /view-model
    savedModelJSON = await model.toJSON();
    savedModelWeights = await model.getWeights().map(w => w.arraySync());
    console.log('--- Modelo guardado en memoria y listo para descargar ---');
    await saveModel(model);
}

//config d multer para el manejo d archivos subidos x el user --> deben ir a uploads
const upload = multer({ dest: 'uploads/'});

//ruta para descargar el modelo en JSON
app.get('/view-model', (req, res) => {
    if(!savedModelJSON || !savedModelWeights) {
        return res.status(404).send('Modelo no encontrado. Entrena el modelo primero :)');
    }
    return res.json({model: savedModelJSON, weights: savedModelWeights});
});


//TODO: optimizar el predict con metodooooo !!!!

app.post('/predict', upload.single('image'), async (req, res) => {
    if(!req.file) {
        return res.status(400).send('Porfa sube una imagen :(');
    }

    try {
        const imgPath = req.file.path; //ruta del archivo temporalment almacenado
        const buffer = await fs.readFile(imgPath); //lectura dl archivo d img

        await fs.unlink(imgPath); //elimina el archivo dspues d procesarlo (para evitar acumulación d archivos)

        //creamos un obj Image con la img subida
        const img = new Image();
        img.src = buffer;

        //creamos un canvas dnd se procesará la img
        const canvas = createCanvas(72, 72);
        const ctx = canvas.getContext('2d');

        //ajustamos el tamaño d la img al esperado x el modelo
        ctx.drawImage(img, 0, 0, 72, 72);

        //convertimos la img en tensor para ser usado en la predicción
        const imgTensor = tf.browser.fromPixels(canvas)
            .toFloat()
            .div(255) //normalizaxión d los valores d pixeles
            .expandDims(); //agrega dimensión para formato esperado x el modelo

        //realiza la prediccion d la img procesada con el modelo cargado
        const prediction = model.predict(imgTensor);
        const predictionValue = (await prediction.data())[0]; //extrae la predicción numerica
        const predictedLabel = predictionValue > 0.5 ? 'Mujer' : 'Hombre'; //interpretamos el resultado en base al value y los label q establecimos

        //devolvemos un json con la predicción
        res.json({ prediction: predictedLabel, confidence: predictionValue});
    } catch(err) {
        console.error('Error en la predicción: ',err);
        res.status(500).send('Error procesando la imagen');
    }
});

//inciamos el server
app.listen(PORT, async () => {
    console.log('--- Iniciando el servidor ---');

    //intentamos cargar un modelo existente
    model = await loadModel();

    if(await modelExists()) {
        console.log('El modelo ya existe y está listo para hacer predicciones');
    } else {
        console.log('Entrenando el modelo porque no se encontraron archivos guardados...');

        //obtenemos los datos d hombres / mujeres d la api
        const maleData = await fetchData(maleUrl);
        const femaleData = await fetchData(femaleUrl);

        //almacenamos en un solo array todos los datos
        let allData = [...maleData, ...femaleData];
        tf.util.shuffle(allData); //y los mezclamos aleatoriamente los datos para evitar sesgos

        //procesamos los datos 
        const trainData = await loadAndProcessImgs(allData);
        await trainModel(model, trainData);
    }

    console.log(`Servidor activo en http://localhost:${PORT}`);
});