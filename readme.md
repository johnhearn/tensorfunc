
Experiments building a functional computation graph:

Experiment 1 (simple evaluation):
```kotlin
        val x = constant(2.0)

        val weights = get_variable("weights", constant(3.0))
        val biases = get_variable("bias", constant(5.0))

        val f = sigma(add(mult(x, weights), biases))
        val y = f() // evaluate

        // Equivalently
        val f = sigma(x * weights + biases)
        val y = f() // evaluate
```

Experiment 2 (functional evaluation):
```kotlin
        val weights = get_variable("weights", tf.constant(3.0))
        val biases = get_variable("bias", tf.constant(5.0))

        val f = sigma(add(mult(weights), biases))

        val x = tf.constant(2.0)
        val y = f(x) // invoke composite function
```

Experiment 3:
```kotlin
        val func = square{square()} // (x^2)^2 = x^4
        assertEquals(2.0 * 2.0 * 2.0 * 2.0, func(2.0)) // invoke composite function

        val composite = square('x')(square('x')) // f: x -> (x^2)^2 = x^4    df/dx=4*x^3
        
        assertEquals(4.0 * 3.0 * 3.0 * 3.0, composite.div('x')(3.0)) // composite functional differentiation
        assertEquals(0.0, composite.div('y')(3.0)) // differentiation with another variable is always 0
```