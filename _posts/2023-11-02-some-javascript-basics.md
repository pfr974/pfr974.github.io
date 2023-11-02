---
layout: post
title:  "Some JavaScript Basics"
date:   2023-11-02
---


# Essential and basic JavaScript concepts

It's time for me to learn Javascript! I wasn't too sure where to start and came across multiple resources. Before following a paid course, I thought I could check what's available for free on Codeacademy and came accross this [JavaScript for Node.js](https://www.codecademy.com/courses/introduction-to-back-end-programming/articles/javascript-for-node-js) article. What's following are some of the notes I took while reading it. The associated github repository can be found [here](https://github.com/pfr974/codeAcademy_intro_back_end_programming_javascript). It's pretty much what's inside the `README.md` file.

## 1] Arrow Expressions

In [`arrow_expressions_demo.js`](https://github.com/pfr974/codeAcademy_intro_back_end_programming_javascript/blob/main/arrow_expressions_demo.js), I first define an **anonymous** arrow function. We do not use the function declaration and simply go with `() => { }`. We can pass arguments to an arrow expression between the parenthesis `(())`.

Following Codeacademy lecture, we could simply write:

```javascript
console.log(() => console.log('Hello, I am an anonymous arrow expression function!'));
```

However, running this code via something like `node arrow_expressions_demo.js` would not print anything to the console. Instead, we would get:

```sh
[Function (anonymous)]
```

 This is a bit confusing at first because we are so used to see examples giving "Hello World!" printed to the console within the first minute of learning about a language.

To actually see this output, we need to invoke the function. We achieve this with:

```javascript
// We define AND invoke an anonymous arrow expression function
(()  => console.log('Hello, I am an anonymous arrow expression function!'))();
```

giving us:

```sh
Hello, I am an anonymous arrow expression function!
```
After this first example, I define a **named** arrow function `namedArrowFunction`:

```javascript
const namedArrowFunction = (name) => {
  console.log(`Hello, my name is ${name}! I am not an anonymous function anymore.`)
};
```

and we invoke it with:
```javascript
namedArrowFunction('Anon');
```

giving us:
```sh
Hello, my name is Anon! I am not an anonymous function anymore.
```

Please note that the use of backticks allows for embedded expressions, i.e. `${name}`. This replaces ``${name}`` with the value of the variable `name` that is passed to the `namedArrowFunction` function. This is a feature of the ES6 version mentioned in the lecture.

## 2] Asynchronous Concept

### 2.1] Promises, start

To illustrate the concept of asynchronous code in the context of Javascript, the lecture decides to introduce us to **Promises**. 

As the name hints at, a promise is an outcome that is not "available" yet, a placeholder. It is the output of an asynchronous operation. In the [`asynchronous_concept_promises.js`](https://github.com/pfr974/codeAcademy_intro_back_end_programming_javascript/blob/main/asynchronous_concept_promises.js) file, we can read:

```javascript
// We create a new promise object and assign it to the testLuck variable.
const testLuck = new Promise((resolve, reject) => {
  if (Math.random() < 0.5) { 
    resolve('Lucky winner!')
  } else {
    reject(new Error('Unlucky!'))
  }
});

testLuck.then(message => {
  console.log(message) // Log the resolved value of the Promise
}).catch(error => {
  console.error(error) // Log the rejected error of the Promise
});
```

A `Promise` can have three different outcomes:
- **pending**, the result is undefined and the expression is waiting for a result;
- **fulfilled**, the promise has been completed successfully and returned a value;
- **rejected**, the promise has been completed unsuccessfully and returned an error object as a result.

In the example provided, a new `Promise` object is created and stored into a `testLuck` constant. It takes a function with two arguments, `resolve` and `reject`. The promise is fulfilled or rejected based on a random generated number being smaller than 0.5 or not. 

The bit after the `testLuck` constant declaration is a chain of two methods, `then` and `catch`. The `then` method is called when the promise is fulfilled, and the `catch` method is called when the promise is rejected.

In the case of a fullfilled promise, the output is:

```sh 
╰─ node asynchronous_concept_promises.js                                                    ─╯
Lucky winner!
```

and in the case of a rejected promise, the output is:

```sh
╰─ node asynchronous_concept_promises.js                                                    ─╯
Error: Unlucky!
    at /Users/pfr974/code/personal/codeAcademy_intro_back_end_programming_javascript/asynchronous_concept_promises.js:6:12
    at new Promise (<anonymous>)
    at Object.<anonymous> (/Users/pfr974/code/personal/codeAcademy_intro_back_end_programming_javascript/asynchronous_concept_promises.js:2:18)
    at Module._compile (internal/modules/cjs/loader.js:1085:14)
    at Object.Module._extensions..js (internal/modules/cjs/loader.js:1114:10)
    at Module.load (internal/modules/cjs/loader.js:950:32)
    at Function.Module._load (internal/modules/cjs/loader.js:790:14)
    at Function.executeUserEntryPoint [as runMain] (internal/modules/run_main.js:76:12)
    at internal/main/run_main_module.js:17:47
```

We should note that this is an abstraction of the concept of asynchronous code and `Promise`. As far as I understand, the code here appears to be executed synchronously/instantaneously due to its relative simplicity. 

We could try to adapt this example by introducing an artifical delay in an alternative version defined in a separate file, e.g. [`asynchronous_concept_promises_alternative.js`](https://github.com/pfr974/codeAcademy_intro_back_end_programming_javascript/blob/main/asynchronous_concept_promises_alternatives.js). 

### 2.2] Promises, slightly more illustrative

The code is the following:
  
  ```javascript
// Creating a new Promise with an artificial delay using setTimeout.
// See https://developer.mozilla.org/en-US/docs/Web/API/setTimeout
const testLuck = new Promise((resolve, reject) => {
  setTimeout(() => {
    if (Math.random() < 0.5) { 
      resolve('Lucky winner!');
    } else {
      reject(new Error('Unlucky!'));
    }
  }, 5000);  // 5000 milliseconds delay, that is 5 seconds
});

console.log('Checking your luck...');  // Log immediately

testLuck.then(message => {
  console.log(message);  // Log the resolved promise output after 5 seconds
}).catch(error => {
  console.error(error);  // Log the rejected promise output after 5 seconds
});

console.log('Waiting for the result...');  // Log immediately as well
  ```

As before, we create a new `Promise` object and store it into a `testLuck` constant. However, the setTimeout method introduces an artificial delay of 5 seconds before deciding if we fulfilled or rejected the promise. The chain of `catch` and `then` methods works the same way as before, with the addition of two extra console logs.

The chain of events is clearer here in my opinion:

1) The first console log is executed immediately
2) A new `Promise` object is created and stored into a constant `testLuck`. Inside, [`setTimeout`](https://developer.mozilla.org/en-US/docs/Web/API/setTimeout) adds an artificial delay of 5 seconds before deciding whether to resolve or reject the promise. 
3) **The second console log is also excuted immediately, not waiting for the promise to be resolved or rejected.**

This gives us the following output:

```sh
node asynchronous_concept_promises_alternatives.js                                                                                                                                                          ─╯
Checking your luck...
Waiting for the result...
``````

4) After 5 seconds, the promise either resolves with:

```sh
─ node asynchronous_concept_promises_alternatives.js                                                                                                                                                          ─╯
Checking your luck...
Waiting for the result...
Lucky winner!
```

or is rejected with:
```sh
─ node asynchronous_concept_promises_alternatives.js                                                                                                                                                          ─╯
Checking your luck...
Waiting for the result...
Error: Unlucky!
    at Timeout._onTimeout (/Users/pfr974/code/personal/codeAcademy_intro_back_end_programming_javascript/asynchronous_concept_promises_alternatives.js:8:14)
    at listOnTimeout (internal/timers.js:557:17)
    at processTimers (internal/timers.js:500:7)
```
