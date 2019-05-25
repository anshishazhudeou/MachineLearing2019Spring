function [] = knn()

% 10 Data Blocks
D = cell(10, 1);
% 10 Label Blocks
L = cell(10, 1);
% Initialize blocks.
for i = 1:10
  dataFileName = strcat('knnData/data', strcat(int2str(i), '.csv'));
  labelsFileName = strcat('knnData/labels', strcat(int2str(i), '.csv'));
  D{i} = csvread(dataFileName);
  L{i} = csvread(labelsFileName);
  [a,b] = size(D{i});
  [c,d] = size(L{i});
  assert(a == 111 && c == 111 && b == 64 && d == 1);
end

% Calculate accuracy for k = 1:30.
for k = 1:30
  x(k) = k;
  y(k) = crossValidation(D, L, k);
end

display(x);
display(y);
% Plot results.
figure(1);
plot(x, y, 'b');
title('Accuracy vs k');
xlabel('r');
ylabel('Accuracy');
hold on;

% Returns the accuracy of 10-fold cross validation on `k` nearest neighbours.
function avgAccuracy = crossValidation(D, L, k)
  result = 0;
  % Split data and labels in ten different ways.
  for i = 1:10
    data = [];
    labels = [];
    for j = 1:10
      if j ~= i
        data = [data; D{j}];
        labels = [labels; L{j}];
      end
    end
    testData = D{i};
    testLabels = L{i};
    acc = applyKNN(data, labels, testData, testLabels, k);
    result = result + acc;
  end
  avgAccuracy = result / 10;

% Returns the accuracy of `k` nearest neighbours.
% Training Set: data & labels.
% Testing Set: testData & testLabels.
function accuracy = applyKNN(data, labels, testData, testLabels, k)
  [m,n] = size(data);
  [p,q] = size(labels);
  [a,b] = size(testData);
  [c,d] = size(testLabels);
  assert(m == p && m == 999);
  assert(a == c && a == 111);
  assert(n == b && n == 64);
  assert(q == d && q == 1);
  testCount = a;
  dataCount = m;
  correct = 0;
  incorrect = 0;

  % For each test:
  for i = 1:testCount
    % For each training:
    for j = 1:dataCount
      e(j) = norm(data(j,:)-testData(i,:));
    end
    [S,I] = sort(e);

    fiveCount = 0;
    sixCount = 0;
    % Tally the k-nearest neighbours.
    for u = 1:k
      num = labels(I(u), 1);
      assert(num == 5 || num == 6);
      if num == 5
        fiveCount = fiveCount + 1;
      else
        sixCount = sixCount + 1;
      end
    end

    % Hypothesis prediction:
    value = tern(fiveCount >= sixCount, 5, 6);
    assert(value == 5 || value == 6);

    % Actual value:
    actualValue = testLabels(i, 1);
    assert(actualValue == 5 || actualValue == 6);

    % Update correct counts.
    if value == actualValue
      correct = correct + 1;
    else
      incorrect = incorrect + 1;
    end
  end

  accuracy = correct / (correct + incorrect);

% Ternary statement.
function r = tern(b, v1, v2)
  if b
    r = v1;
  else
    r = v2;
  end
