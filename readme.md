
Experiments building a functional computation graph:

Example 1:
```
        val x = constant(2.0)

        val weights = get_variable("weights", constant(3.0))
        val biases = get_variable("bias", constant(5.0))

        val f = sigma(add(mult(x, weights), biases))
        val y = f()

        // Equivalently
        val f = sigma(x * weights + biases)
        val y = f()
```

Example 2:
```
        val weights = get_variable("weights", tf.constant(3.0))
        val biases = get_variable("bias", tf.constant(5.0))

        val f = sigma(add(mult(weights), biases))

        val x = tf.constant(2.0)
        val y = f(x)
```
