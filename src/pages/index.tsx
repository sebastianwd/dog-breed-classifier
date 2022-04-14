/* eslint-disable @next/next/no-img-element */
import type { NextPage } from 'next'
import React, { useState, useRef, useReducer } from 'react'
// import * as mobilenet from '@tensorflow-models/mobilenet'
import * as tf from '@tensorflow/tfjs'
import { noop, upperFirst } from 'lodash'
import { classnames } from '~/classnames'

const states = {
  initial: { on: { next: 'loadingModel' } },
  loadingModel: { on: { next: 'modelReady' } },
  modelReady: { on: { next: 'imageReady' } },
  imageReady: { on: { next: 'identifying' }, showImage: true },
  identifying: { on: { next: 'complete' } },
  complete: { on: { next: 'modelReady' }, showImage: true, showResults: true },
} as const

type States = {
  [key in keyof typeof states]: { on: { next: keyof typeof states }; showImage?: boolean; showResults?: boolean }
}

const machine: { initial: 'initial'; states: States } = {
  initial: 'initial',
  states,
} as const

type ClassificationResults =
  | {
      className: string
      probability: number
    }[]
  | undefined

tf.setBackend('cpu')

const Home: NextPage = () => {
  const [results, setResults] = useState<ClassificationResults>([])
  const [imageURL, setImageURL] = useState<string>()

  const [graphModel, setgraphModel] = useState<tf.GraphModel | null>(null)

  const imageRef = useRef<HTMLImageElement | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)

  const reducer = (state: keyof States, event: 'next'): keyof States =>
    machine.states[state].on[event] || machine.initial

  const [appState, dispatch] = useReducer(reducer, machine.initial)

  const next = () => dispatch('next')

  const loadModel = async () => {
    next()

    const graphModel = await tf.loadGraphModel('model/model.json')

    // const model = await mobilenet.load()

    setgraphModel(graphModel)
    next()
  }

  const identify = async () => {
    next()

    const image = tf.browser.fromPixels(imageRef.current!).resizeNearestNeighbor([180, 180]).toFloat().expandDims(0)

    const predictions = (await tf.tidy(() => graphModel!.predict(image))) as tf.Tensor<tf.Rank>

    const probabilities = tf.softmax(predictions)

    const predictionTypedArray = Array.from(probabilities.dataSync())

    /* 
    const bestIndex = tf.argMax(predictionTypedArray).dataSync()
    const bestProbability = tf.max(predictionTypedArray).dataSync()

    console.log('best result: ', classnames[bestIndex], 100 * bestProbability)
    */

    const results = Array.from(predictionTypedArray)
      .map((p, i) => {
        return {
          probability: p,
          className: classnames[i],
        }
      })
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 5)

    // const results = imageRef.current && (await model?.classify(imageRef.current))

    setResults(results || [])

    next()
  }

  const reset = async () => {
    setResults([])
    next()
  }

  const upload = () => inputRef.current && inputRef.current.click()

  const handleUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { files } = event.target

    if (files && files.length > 0) {
      const url = URL.createObjectURL(files[0])

      setImageURL(url)

      next()
    }
  }

  const actionButton = {
    initial: { action: loadModel, text: 'Load Model' },
    loadingModel: { action: noop, text: 'Loading Model...' },
    modelReady: { action: upload, text: 'Upload Image' },
    imageReady: { action: identify, text: 'Identify Breed' },
    identifying: { action: noop, text: 'Identifying...' },
    complete: { action: reset, text: 'Reset' },
  }

  const { showImage, showResults } = machine.states[appState]

  return (
    <div>
      {showImage && <img src={imageURL} alt='upload-preview' ref={imageRef} />}
      <input type='file' accept='image/*' capture='environment' onChange={handleUpload} ref={inputRef} />
      {showResults && (
        <ul>
          {results?.map(({ className, probability }) => (
            <li key={className}>{`${upperFirst(
              className.substring(className.indexOf('-') + 1).replaceAll('_', ' '),
            )}: ${(probability * 100).toFixed(2)}%`}</li>
          ))}
        </ul>
      )}
      <button onClick={actionButton[appState].action}>{actionButton[appState].text}</button>
    </div>
  )
}

export default Home
