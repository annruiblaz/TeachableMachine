const express = require('express'); //para manejar peticiones http
const axios = require('axios'); //nos permite hacer las solicitudes http
const tf = require('@tensorflow/tfjs'); //para trabajar con modelos d aprendizaje profundo en JS
const {Image, createCanvas} = require('canvas'); //nos permite manipular imgs en Node.js

//creamos instancia d app express y definimos el puerto
const app = express();
const PORT = 3000;

//habilitamos el manejo d json en las solicitudes
app.use(express.json());

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

//función para entrenar el modelo con los datos ya procesados
async function trainModel(model, trainData) {
    await model.fit(trainData.xs, trainData.ys, {
        epochs: 16, //num d épocas de entrenamiento
        batchSize: 64, //tamaño del lote
        shuffle: true, //mezclar los datos para mejorar el entrenamiento
        validationSplit: 0.2 //porcentaje d datos utilizados para la validación
    });

    console.log('--- Entrenamiento completado ---');
}

//función para hacer predicciones con imgs nuevas
async function predict(model, testData) {
    const predictions = [];
    for(const item of testData) {
        const response = await axios.get(item.image, { responseType: 'arraybuffer'});
        //el buffer es para optimizar la lectura d bytes
        const buffer = Buffer.from(response.data);
        const img = new Image();
        img.src = buffer;
        const canvas = createCanvas(72, 72);
        const ctx = canvas.getContext('2d');

        ctx.drawImage(img, 0, 0, 72, 72);

        const imgTensor = tf.browser.fromPixels(canvas)
            .toFloat()
            .div(255)
            .expandDims(); //agrega dimensión d batch

        const prediction = model.predict(imgTensor);
        const predictionValue = (await prediction.data())[0]; //extrae la predicción numerica
        const predictedLabel = predictionValue > 0.5 ? 'Mujer' : 'Hombre'; //interpretamos el resultado en base al value y los label q establecimos

        predictions.push({
            image: item.image,
            prediction: predictedLabel,
            confidence: predictionValue //guardamos el valor crudo d la predicción
        });
    }
    console.log('Predictions: ', predictions);
}

//función principal para entrenar y probar el modelo
async function main() {
    //obtenemos los datos d hombres / mujeres d la api
    const maleData = await fetchData(maleUrl);
    const femaleData = await fetchData(femaleUrl);

    //almacenamos en un solo array todos los datos
    let allData = [...maleData, ...femaleData];
    tf.util.shuffle(allData); //y los mezclamos aleatoriamente los datos para evitar sesgos

    //procesamos y almacenamos los datos (imgs) 
    const trainData = await loadAndProcessImgs(allData);
    const model = createModel();//creamos el modelo

    //entrenamos el modelo con los datos procesados
    await trainModel(model, trainData);
    await predict(model, allData.slice(0, 10)); // y realizamos las predicciones con los 10 primeros datos del array 
}

//ejecutamos el metodo
main();