import numpy as np
import tensorflow as tf

class PPConv3D( tf.keras.layers.Layer ) :  # implements push-pull (surround suppression) convolution layer

  def __init__( self, out_channels, kernel_size, surround_type = 'scale', strides = 1, padding='SAME', dilation = 1,
                kernel_initializer = 'glorot_uniform', alpha = 1, scale = 3, **kwargs ) :
      super( PPConv3D, self).__init__()

      if alpha :  # alpha is not None
          self.alpha = alpha
      else :   # trainable alpha with initial value 1
          self.alpha = tf.Variable( 1., name = 'alpha', trainable = True, aggregation = tf.VariableAggregation.MEAN )
      self.scale_factor = scale
      self.surround_type = surround_type
      self.padding = padding
      self.push_size = np.array( kernel_size ).astype( int )
      self.push_channels = out_channels
      self.strides = strides
      self.dilation = dilation
      self.kernel_initializer = kernel_initializer

      # compute the size of the pull kernel
      if self.scale_factor == 1 :
          self.pull_size = self.push_size
      else :
          self.pull_size = np.floor( self.push_size * scale ).astype( int )

  def get_config( self ) :  # this function is needed to save the layer when saving model or checkpoints
    config = super( PPConv3D, self ).get_config()
    config.update( { 'out_channels' : self.push_channels } )
    config.update( { 'kernel_size' : self.push_size } )
    config.update( { 'surround_type' : self.surround_type } )
    config.update( { 'strides' : self.strides } )
    config.update( { 'padding' : self.padding } )
    config.update( { 'dilation' : self.dilation } )
    config.update( { 'kernel_initializer' : self.kernel_initializer } )
    config.update( { 'alpha' : self.alpha } )
    config.update( { 'scale' : self.scale_factor } )
    return config

  def build( self, input_shape ) :
      # Create trainable push weights
      push_filt_shape = np.concatenate( ( self.push_size, [ input_shape[ -1 ], self.push_channels ] ) ).astype( int )
      self.push_weights = self.add_weight( name = 'push_weights', shape = push_filt_shape,
                                           initializer = self.kernel_initializer, trainable = True, aggregation = tf.VariableAggregation.MEAN )
      # Create a non-trainable pull weights
      pull_filt_shape = np.concatenate( ( self.pull_size, [ input_shape[ -1 ], self.push_channels ] ) ).astype( int )
      self.pull_weights = self.add_weight( name = 'pull_weights', shape = pull_filt_shape,
                                          initializer = 'zeros', trainable = False, aggregation = tf.VariableAggregation.MEAN )

      if self.surround_type == 'scale' :  # Create bilinear upsampling weights matrix
          filter_size = 2 * self.scale_factor - self.scale_factor % 2
          upsample_filter = np.zeros( ( filter_size,
                                        filter_size,
                                        filter_size,
                                        input_shape[ -1 ],
                                        input_shape[ -1 ] ), dtype = np.float32 )
          if filter_size % 2 == 1 :
              center = self.scale_factor - 1
          else :
              center = self.scale_factor - 0.5
          og = np.ogrid[ : filter_size, : filter_size, : filter_size ]  # makes grid vectors along DHW dimensions
          upsample_kernel =  ( 1 - abs( og[ 0 ] - center ) / self.scale_factor ) * \
                             ( 1 - abs( og[ 1 ] - center ) / self.scale_factor ) * \
                             ( 1 - abs( og[ 2 ] - center ) / self.scale_factor )  # tensor product of the 3 centered grid vector
      else :  # tile
          filter_size = self.scale_factor
          upsample_filter = np.zeros( ( filter_size,
                                        filter_size,
                                        filter_size,
                                        input_shape[ -1 ],
                                        input_shape[ -1 ] ), dtype = np.float32 )
          upsample_kernel = np.ones( ( 3, 3, 3 ) ) / self.scale_factor**3
      for i in range( input_shape[ -1 ] ) :
          upsample_filter[ :, :, :, i, i ] = upsample_kernel
      self.upsample_filter = tf.constant( upsample_filter, dtype = 'float32', name = 'upsample_filter' )  # store as tf constant


  def call( self, x ) :
      # up-sample the pull kernel from the push kernel, if necessary
      if self.scale_factor == 1 :  # nothing to be done, just copy push weights
          self.pull_weights = self.push.weights
      else :
          if self.surround_type == 'scale' :
              # up-sample the push_weights tensor using the bilinear upsampling kernel build in build() and the tf.conv3d_transpose()
              tmp = tf.transpose( self.push_weights, perm = [ 4, 0, 1, 2, 3 ] ) # permute to NDHWC arrangement to up-sample DHW dimensions only in conv3d_transpose
              # tmp = upsample_tf( tmp, self.scale_factor, self.upsample_filter )
              tmp = tf.nn.conv3d_transpose( tmp, self.upsample_filter,
                                            output_shape = [ tmp.shape[ 0 ],
                                                             tmp.shape[ 1 ] * self.scale_factor,
                                                             tmp.shape[ 2 ] * self.scale_factor,
                                                             tmp.shape[ 3 ] * self.scale_factor,
                                                             tmp.shape[ 4 ] ],
                                            strides = [ 1, self.scale_factor, self.scale_factor, self.scale_factor, 1 ],
                                            name = 'UpsampleKernel' + str( self.scale_factor ) )
              tmp = tf.transpose( tmp, perm = [ 1, 2, 3, 4, 0 ] )  # permute the result back to normal DHWCC kernel arrangement
          else :  # tile the push_weights kernel 3-fold in each dimension
              tmp = tf.transpose( self.upsample_filter, perm = [ 4, 0, 1, 2, 3 ] )  # permute to NDHWC arrangement to up-sample DHW dimensions only in conv3d_transpose
              tmp = tf.nn.conv3d_transpose( tmp, tf.transpose( self.push_weights, perm = [ 0, 1, 2, 4, 3 ] ),
                                            output_shape = [ self.push_weights.shape[ 3 ],
                                                             self.push_weights.shape[ 0 ] * self.scale_factor,
                                                             self.push_weights.shape[ 1 ] * self.scale_factor,
                                                             self.push_weights.shape[ 2 ] * self.scale_factor,
                                                             self.push_weights.shape[ 4 ] ],
                                            strides = [ 1, self.scale_factor, self.scale_factor, self.scale_factor, 1 ],
                                            name = 'TileKernel' + str( self.scale_factor ) )
              tmp = tf.transpose( tmp, perm = [ 1, 2, 3, 0, 4 ] )  # permute the result back to normal DHWCC kernel arrangement
          self.pull_weights.assign( tmp )  # update the pull weights

      # return the combined push-pull convolution: push convolution with '+' sign and pull convolution with '-' sign
      push = tf.nn.conv3d( x, filters = self.push_weights, strides = [ 1, self.strides, self.strides, self.strides, 1 ], padding = self.padding )
      pull = tf.nn.conv3d( x, filters = self.pull_weights, strides = [ 1, self.strides, self.strides, self.strides, 1 ], padding = self.padding )
      return tf.add( tf.nn.relu( push ), -self.alpha * tf.nn.relu( pull ), name = 'PushPullConv3D' )
