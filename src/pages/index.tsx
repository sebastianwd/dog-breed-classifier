/* eslint-disable @next/next/no-img-element */
import type { NextPage } from 'next'
import React, { useState, useRef, useReducer, useEffect } from 'react'
// import * as mobilenet from '@tensorflow-models/mobilenet'
import * as tf from '@tensorflow/tfjs'
import { noop, upperFirst } from 'lodash'
import { classnames } from '~/classnames'
import {
  Box,
  Button,
  Container,
  Heading,
  Stack,
  Table,
  TableContainer,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from '@chakra-ui/react'

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
  const [imageURL, setImageURL] = useState<string>('/dog-breed-placeholder.png')

  const [graphModel, setgraphModel] = useState<tf.GraphModel | null>(null)

  const imageRef = useRef<HTMLImageElement | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)

  const reducer = (state: keyof States, event: 'next'): keyof States =>
    machine.states[state].on[event] || machine.initial

  const [appState, dispatch] = useReducer(reducer, machine.initial)

  const next = () => dispatch('next')

  const loadModel = React.useCallback(async () => {
    next()

    const graphModel = await tf.loadGraphModel('model/model.json')

    // const model = await mobilenet.load()

    setgraphModel(graphModel)
  }, [])

  useEffect(() => {
    if (appState === 'initial') {
      loadModel()
    }
  }, [appState, loadModel])

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

    if (inputRef.current) {
      inputRef.current.value = ''

      setImageURL('/dog-breed-placeholder.png')
    }

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
    initial: { action: noop, text: 'Cargando Modelo...' },
    loadingModel: { action: noop, text: 'Cargando Modelo...' },
    modelReady: { action: upload, text: 'Subir Imagen' },
    imageReady: { action: identify, text: 'Identificar Raza' },
    identifying: { action: noop, text: 'Identificando...' },
    complete: { action: reset, text: 'Subir Otra Imagen' },
  }

  const { showImage, showResults } = machine.states[appState]

  return (
    <Container maxW='md' color='white' py='18' centerContent>
      <img src={imageURL} alt='upload-preview' ref={imageRef} />
      <input
        type='file'
        accept='image/*'
        capture='environment'
        style={{ display: 'none' }}
        onChange={handleUpload}
        ref={inputRef}
      />
      <Box
        backgroundColor='gray.700'
        borderRadius={8}
        boxShadow='0px 4px 10px rgba(0, 0, 0, 0.05)'
        py={8}
        px={8}
        width='100%'>
        <Stack justifyContent='center' spacing={8} borderRadius={8}>
          <Button
            backgroundColor='gray.900'
            fontWeight='medium'
            mt={4}
            _hover={{ bg: 'gray.800' }}
            _active={{
              bg: 'gray.600',
              transform: 'scale(0.95)',
            }}
            isLoading={['identifying', 'loadingModel'].includes(appState)}
            disabled={['initial'].includes(appState)}
            onClick={actionButton[appState].action}>
            {actionButton[appState].text}
          </Button>
          {showResults ? (
            <TableContainer>
              <Table variant='simple'>
                <Thead>
                  <Tr>
                    <Th>Raza</Th>
                    <Th isNumeric>Precisión</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {results?.map(({ className, probability }) => (
                    <Tr key={className}>
                      <Td>{upperFirst(className.substring(className.indexOf('-') + 1).replaceAll('_', ' '))}</Td>
                      <Td isNumeric>{(probability * 100).toFixed(2)}%</Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            </TableContainer>
          ) : (
            <EmptyState />
          )}
        </Stack>
      </Box>
    </Container>
  )
}

const EmptyState = () => {
  return (
    <Heading size='md' textAlign='center'>
      Sube la imagen de tu mascota para identificar su raza.
    </Heading>
  )
}

export default Home
