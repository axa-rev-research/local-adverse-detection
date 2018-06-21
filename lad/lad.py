import numpy, pandas
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import brute
from scipy.spatial.distance import euclidean

# to plot dtree
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus

class LocalSurrogate():
    
    def __init__(self, X, n_support_points=100, n_segment_points=100, blackbox=None, max_depth=5):
        
        self.support_points_min = X.min()
        self.support_points_max = X.max()
        self.n_support_points = n_support_points
        self.n_segment_points = n_segment_points
        self.blackbox = blackbox
        self.touchpoint_hypersphere_radius = 0.1
        self.touchpoint_hypersphere_n_points = 10
        self.max_depth = max_depth
    
    
    def get_support_points(self, x_toexplain):
        
        support_points = []
        
        while len(support_points) < self.n_support_points:
            candidate_ = numpy.random.uniform(low=self.support_points_min,
                                       high=self.support_points_max)
            
            if self.blackbox.predict([candidate_]) != self.blackbox.predict([x_toexplain]):
                support_points.append(candidate_)
        
        support_points = pandas.DataFrame(support_points, columns=x_toexplain.index)
        
        return support_points

    
    def get_segment_points(self, x_toexplain, support_point):
        
        ## Draw random points on the segment [x_toexplain ; support point]
        linear_model = LinearRegression()
        df = pandas.concat((x_toexplain, support_point), axis=1).T
        x = df.iloc[:,[0]].values
        y = df.iloc[:,1:].values
        linear_model.fit(x,y)

        segment_points_x = numpy.random.uniform(low=x.min(), high=x.max(), size=self.n_segment_points)
        segment_points_x.shape = (len(segment_points_x),1)
        segment_points_y = linear_model.predict(segment_points_x)
        segment_points = pandas.DataFrame(numpy.concatenate((segment_points_x, segment_points_y), axis=1))

        return segment_points, linear_model


    def get_segment_boundary_touchpoint(self, x_toexplain, support_point, segment_points, linear_model):
        
        def optim_objective(x, *args):
            segment_points_x, segment_points_labels = args
            segment_points_x = segment_points_x.values.reshape(len(segment_points_x),)

            ## /!\ /!\ Issue when more than 2 classes ?
            labels = (segment_points_x < x[0])*1
            score = normalized_mutual_info_score(segment_points_labels, labels.reshape(len(labels),))
            return (1-score)
            
        ## Label each drawn points on the segment
        segment_points_labels = self.blackbox.predict(segment_points)
        
        ## Find the min of the information gain (or ~ measure to get the frontier frontier touchpoint)
        j = 0 # /!\ /!\ j=0 parce qu'on considère que c'est le x => à uniformiser / locker
        res = brute(optim_objective,
                    ranges=[(min(x_toexplain.iloc[j],
                    support_point.iloc[j]),
                    max(x_toexplain.iloc[j],
                    support_point.iloc[j]))],
                    args=(segment_points.iloc[:,[j]], segment_points_labels))
        
        # Once we got the x for the touchpoint, we compute the rest of the coordinates
        touchpoint_x = numpy.array([res])
        touchpoint_x.shape = (len(touchpoint_x),1)
        touchpoint_y = linear_model.predict(touchpoint_x)
        touchpoint = numpy.concatenate((touchpoint_x, touchpoint_y), axis=1)
        
        return pandas.DataFrame(touchpoint), segment_points_labels


    def get_random_points_hypersphere_touchpoint(self, touchpoint):

        radius_ = self.touchpoint_hypersphere_radius
        n_points_ = self.touchpoint_hypersphere_n_points

        res = []
        while len(res) < n_points_:
        
            n_points_left_ = n_points_ - len(res)
            # About half the points are lost in the test hypercube => hypersphere
            lbound = numpy.repeat(touchpoint.values-(radius_/2.), n_points_left_*2, axis=0)
            hbound = numpy.repeat(touchpoint.values+(radius_/2.), n_points_left_*2, axis=0)
            points = numpy.random.uniform(low=lbound, high=hbound)
            
            # Check if x_generated is within hypersphere (if kind=='hypersphere')
            for x_generated in points:
                if euclidean(x_generated, touchpoint.values) < radius_:
                    res.append(x_generated)
                if len(res) == n_points_:
                    break

        return pandas.DataFrame(numpy.array(res))


    def compute_surrogate(self, support_points, plot=False):
        
        X_surrogate = []
        for k in support_points.keys():
            X_surrogate.append(support_points[k]['touchpoint'])
            X_surrogate.append( support_points[k]['touchpoint_hypersphere_points'] )
        X_surrogate = pandas.DataFrame(numpy.concatenate(X_surrogate, axis=0))
        y_surrogate = self.blackbox.predict(X_surrogate)
        
        surrogate = DecisionTreeClassifier(max_depth=self.max_depth).fit(X_surrogate, y_surrogate)
        
        if plot:
            xx, yy = np.mgrid[X_surrogate.min().iloc[0]:X_surrogate.max().iloc[0]:.01, X_surrogate.min().iloc[1]:X_surrogate.max().iloc[1]:.01]
            grid = np.c_[xx.ravel(), yy.ravel()]
            probs = surrogate.predict_proba(grid)[:, 1].reshape(xx.shape)
            surrogate['xx'] = xx
            surrogate['yy'] = yy
            surrogate['probs'] = probs
                
        return surrogate


    def get_local_surrogate(self, x_toexplain):

        support_points_ = self.get_support_points(x_toexplain)
        support_points = {}

        for i in range(self.n_support_points):
            support_point = support_points_.iloc[i]
            segment_points, linear_model = self.get_segment_points(x_toexplain,
                                                                support_point)

            touchpoint, segment_points_labels = self.get_segment_boundary_touchpoint(x_toexplain,
                                                                                     support_point,
                                                                                     segment_points,
                                                                                     linear_model)

            touchpoint_hypersphere_points = self.get_random_points_hypersphere_touchpoint(touchpoint)


            support_points[i] = {
                    'support_point':support_point,
                    'segment_points':segment_points,
                    'segment_points_labels':segment_points_labels,
                    'touchpoint':touchpoint,
                    'linear_model':linear_model,
                    'touchpoint_hypersphere_points':touchpoint_hypersphere_points
                }

        surrogate = self.compute_surrogate(support_points)
        
        return support_points, surrogate

    
    def plot_support_points(self, X, x_toexplain, support_points, ax=None,
                            plot_support_points=True,
                            plot_segment_points=True,
                            plot_touchpoints=True,
                            plot_touchpoint_hypersphere_points=False):

        colors = sns.color_palette()
        if ax is None:
            f, ax = subplots(1)

        plot_classification_contour(X, self.blackbox, ax)

        for i in range(self.n_support_points):
            support_point = support_points[i]
            # /!\ only works for 2 classes problems
            
            if plot_segment_points:
                support_point['segment_points'][support_point['segment_points_labels']==0].plot(kind='scatter', x=0, y=1, ax=ax, c=colors[3], marker='.')
                support_point['segment_points'][support_point['segment_points_labels']==1].plot(kind='scatter', x=0, y=1, ax=ax, c=colors[0], marker='.')
            if plot_support_points:
                plot(support_point['support_point'][0], support_point['support_point'][1], 'kx')
            
            if plot_touchpoint_hypersphere_points:
                for k in support_points.keys():
                    tmp = support_points[k]['touchpoint_hypersphere_points']
                    plot(tmp[:,0], tmp[:,1],'.', color=colors[2])
            if plot_touchpoints:
                plot(support_point['touchpoint'][0], support_point['touchpoint'][1], c=colors[8], marker='o')
            
            plot(x_toexplain[0], x_toexplain[1], 'rp')
            
    def plot_decision_tree(self, X, surrogate):
        
        dot_data = StringIO()
        export_graphviz(surrogate, out_file=dot_data,  
                        filled=True, rounded=True,
                        special_characters=True, feature_names = X.columns)

        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        
        return graph.create_png()
